from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .features import (
    ACTION_DIM,
    ENTITY_DIM,
    HISTORY_DIM,
    MAX_DECK_SIZE,
    STATE_DIM,
    action_features,
    history_features,
    state_features,
    structured_observation_features,
)
from .model import MaskedPointerActorCritic, legal_choice_mask


SPLIT_SEED = 20260720
VALIDATION_FRACTION = 0.20
FORBIDDEN_MODEL_FIELDS = {
    "winner", "outcome", "policy_weight", "value_weight", "source_path", "source_sha256",
    "action_status", "observation.logs", "action",
}
MODEL_INPUT_FIELDS = {
    "observation.current.visible_state",
    "observation.select.type",
    "observation.select.minCount",
    "observation.select.maxCount",
    "observation.select.option",
}
HISTORY_MODEL_INPUT_FIELDS = MODEL_INPUT_FIELDS | {
    "history.prior_pre_action_observation.current.visible_state",
    "history.prior_selected_options",
}
STRUCTURED_MODEL_INPUT_FIELDS = MODEL_INPUT_FIELDS | {
    "observation.current.visible_card_entities",
    "observation.select.visible_card_entities",
    "observation.select.option.card_attack_identity",
    "episode.acting_player_submitted_deck",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compact_replay_row(raw: dict[str, Any], structured: bool = False) -> dict[str, Any]:
    """Validate one schema-v2 row and retain only compact pre-action inputs plus labels."""

    if raw.get("schema_version") != 2:
        raise ValueError("schema_version must be 2")
    observation = raw.get("observation")
    if not isinstance(observation, dict) or "logs" in observation:
        raise ValueError("observation must be an object without logs")
    select = observation.get("select")
    if not isinstance(select, dict):
        raise ValueError("select must be an object")
    options = select.get("option")
    action = raw.get("action")
    if not isinstance(options, list) or not isinstance(action, list):
        raise ValueError("options and action must be lists")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in action):
        raise ValueError("action must contain integer indices")
    min_count = int(select.get("minCount", 0))
    max_count = int(select.get("maxCount", min_count))
    if min_count < 0 or max_count < min_count:
        raise ValueError("invalid selection bounds")
    if not min_count <= len(action) <= max_count:
        raise ValueError("action length violates selection bounds")
    if len(set(action)) != len(action):
        raise ValueError("action contains duplicate indices")
    if any(index < 0 or index >= len(options) for index in action):
        raise ValueError("action index outside option range")
    policy_weight = float(raw.get("policy_weight"))
    value_weight = float(raw.get("value_weight"))
    outcome = float(raw.get("outcome"))
    if policy_weight not in (0.0, 1.0) or value_weight != 1.0:
        raise ValueError("unexpected policy/value weight")
    if policy_weight != float(outcome > 0):
        raise ValueError("policy weight must select winner rows only")
    if not math.isfinite(outcome):
        raise ValueError("non-finite outcome")
    encoded_options = [action_features(option if isinstance(option, dict) else {}, index) for index, option in enumerate(options)]
    row = {
        "episode_id": str(raw.get("episode_id")),
        "player": raw.get("player"),
        "action_step": raw.get("action_step"),
        "observation_step": raw.get("observation_step"),
        "state": state_features(observation),
        "options": encoded_options,
        "action": list(action),
        "min_count": min_count,
        "max_count": max_count,
        "outcome": outcome,
        "policy_weight": policy_weight,
        "value_weight": value_weight,
        "select_type": select.get("type"),
        "select_context": _context_key(select.get("context")),
        "source_sha256": str(raw.get("source_sha256")),
    }
    if structured:
        row.update(structured_observation_features(observation, options))
    return row


def build_causal_histories(rows: list[dict[str, Any]], max_history: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach prior same-player decisions using explicit replay keys, never file order."""

    if max_history <= 0:
        raise ValueError("history max length must be positive")
    prepared = [dict(row) for row in rows]
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in prepared:
        player = row.get("player")
        action_step = row.get("action_step")
        observation_step = row.get("observation_step")
        if not isinstance(player, int) or isinstance(player, bool):
            raise ValueError("history requires integer player identity")
        if not isinstance(action_step, int) or isinstance(action_step, bool):
            raise ValueError("history requires integer action_step")
        if not isinstance(observation_step, int) or isinstance(observation_step, bool):
            raise ValueError("history requires integer observation_step")
        if observation_step >= action_step:
            raise ValueError("history requires a pre-action observation_step")
        groups[(row["episode_id"], player)].append(row)

    history_rows = history_tokens = truncated_rows = 0
    for key, group in groups.items():
        group.sort(key=lambda row: row["action_step"])
        steps = [row["action_step"] for row in group]
        if len(steps) != len(set(steps)):
            raise ValueError(f"duplicate history action_step for episode/player {key}")
        prior: list[tuple[int, list[float]]] = []
        for row in group:
            visible = prior[-max_history:]
            row["history"] = [token for _, token in visible]
            row["history_steps"] = [step for step, _ in visible]
            if any(step >= row["action_step"] for step in row["history_steps"]):
                raise AssertionError("non-causal history token")
            history_rows += int(bool(visible))
            history_tokens += len(visible)
            truncated_rows += int(len(prior) > max_history)
            prior.append((row["action_step"], history_features(row["state"], row["options"], row["action"])))
    audit = {
        "enabled": True,
        "max_length": max_history,
        "group_by": ["episode_id", "player"],
        "order_by": "action_step",
        "groups": len(groups),
        "rows_with_history": history_rows,
        "history_tokens": history_tokens,
        "truncated_rows": truncated_rows,
        "current_or_future_steps_used": 0,
        "physical_row_order_used": False,
    }
    return prepared, audit


def load_deck_map(path: Path) -> tuple[dict[tuple[str, int], list[int]], dict[str, Any]]:
    result: dict[tuple[str, int], list[int]] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = json.loads(line)
            episode_id, player = str(raw.get("episode_id")), raw.get("player")
            deck = raw.get("deck")
            if not isinstance(player, int) or isinstance(player, bool) or player not in (0, 1):
                raise ValueError(f"invalid deck-map player on line {line_number}")
            if not isinstance(deck, list) or len(deck) != MAX_DECK_SIZE:
                raise ValueError(f"invalid deck-map deck on line {line_number}")
            if any(not isinstance(card_id, int) or isinstance(card_id, bool) for card_id in deck):
                raise ValueError(f"invalid deck-map card ID on line {line_number}")
            key = (episode_id, player)
            if key in result and result[key] != deck:
                raise ValueError(f"conflicting deck map entry for {key}")
            result[key] = deck
    return result, {
        "path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
        "entries": len(result), "episodes": len({key[0] for key in result}),
    }


def load_replay_dataset(
    path: Path,
    history_length: int = 0,
    deck_map_path: Path | None = None,
    structured: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if structured and deck_map_path is None:
        raise ValueError("structured dataset requires a deck map")
    deck_map: dict[tuple[str, int], list[int]] = {}
    deck_audit: dict[str, Any] = {"enabled": False}
    if deck_map_path is not None:
        deck_map, deck_audit = load_deck_map(deck_map_path)
        deck_audit["enabled"] = True
    rows: list[dict[str, Any]] = []
    empty = multi = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = compact_replay_row(json.loads(line), structured=structured)
            except Exception as exc:
                raise ValueError(f"invalid replay row {line_number}: {exc}") from exc
            if structured:
                player = row.get("player")
                key = (row["episode_id"], player)
                if key not in deck_map:
                    raise ValueError(f"missing acting-player deck for {key}")
                # Reuse the sidecar list object across an episode/player instead
                # of duplicating 60 Python integers for every decision row.
                row["deck_card_ids"] = deck_map[key]
            rows.append(row)
            empty += int(len(row["action"]) == 0)
            multi += int(len(row["action"]) > 1)
    history_audit: dict[str, Any] = {"enabled": False, "max_length": 0}
    if history_length:
        rows, history_audit = build_causal_histories(rows, history_length)
    else:
        for row in rows:
            row["history"] = []
            row["history_steps"] = []
    episodes = {row["episode_id"] for row in rows}
    model_input_fields = (
        STRUCTURED_MODEL_INPUT_FIELDS if structured else
        HISTORY_MODEL_INPUT_FIELDS if history_length else MODEL_INPUT_FIELDS
    )
    audit = {
        "input": str(path),
        "input_sha256": sha256_file(path),
        "input_bytes": path.stat().st_size,
        "readable_rows": len(rows),
        "episodes": len(episodes),
        "policy_rows": sum(int(row["policy_weight"] == 1.0) for row in rows),
        "value_rows": sum(int(row["value_weight"] == 1.0) for row in rows),
        "empty_action_rows": empty,
        "multi_select_rows": multi,
        "unsupported_rows": 0,
        "skipped_rows": 0,
        "invalid_rows": 0,
        "schema_version": 2,
        "observation_logs_rows": 0,
        "model_input_fields": sorted(model_input_fields),
        "forbidden_model_fields_used": sorted(model_input_fields & FORBIDDEN_MODEL_FIELDS),
        "history": history_audit,
        "structured": structured,
        "deck_map": deck_audit,
    }
    return rows, audit


def split_episode_ids(episode_ids: Iterable[str], validation_fraction: float = VALIDATION_FRACTION, seed: int = SPLIT_SEED) -> tuple[set[str], set[str]]:
    episodes = sorted({str(value) for value in episode_ids})
    if not episodes:
        return set(), set()
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    shuffled = list(episodes)
    random.Random(seed).shuffle(shuffled)
    validation_count = 0 if validation_fraction == 0 or len(episodes) < 2 else max(1, min(len(episodes) - 1, round(len(episodes) * validation_fraction)))
    validation = set(shuffled[:validation_count])
    return set(episodes) - validation, validation


class TrajectoryDataset(Dataset):
    """All validated rows; no action cardinality or select context is filtered."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _episode_id(row: dict[str, Any]) -> str:
    return str(row.get("episode_id", row.get("episode")))


def split_by_episode(dataset: TrajectoryDataset, validation_fraction: float, seed: int) -> tuple[TrajectoryDataset, TrajectoryDataset]:
    train_ids, validation_ids = split_episode_ids((_episode_id(row) for row in dataset.rows), validation_fraction, seed)
    train = [row for row in dataset.rows if _episode_id(row) in train_ids]
    validation = [row for row in dataset.rows if _episode_id(row) in validation_ids]
    return TrajectoryDataset(train), TrajectoryDataset(validation)


def build_split_manifest(rows: list[dict[str, Any]], validation_fraction: float = VALIDATION_FRACTION, seed: int = SPLIT_SEED) -> tuple[dict[str, Any], set[str], set[str]]:
    train_ids, validation_ids = split_episode_ids((row["episode_id"] for row in rows), validation_fraction, seed)
    per_episode: dict[str, dict[str, Any]] = {}
    for row in rows:
        episode_id = row["episode_id"]
        record = per_episode.setdefault(episode_id, {
            "episode_id": episode_id,
            "split": "validation" if episode_id in validation_ids else "train",
            "row_count": 0,
            "policy_row_count": 0,
            "value_row_count": 0,
            "source_sha256": row["source_sha256"],
        })
        if record["source_sha256"] != row["source_sha256"]:
            raise ValueError(f"conflicting source SHA within episode {episode_id}")
        record["row_count"] += 1
        record["policy_row_count"] += int(row["policy_weight"] == 1.0)
        record["value_row_count"] += int(row["value_weight"] == 1.0)
    manifest = {
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "train_episodes": len(train_ids),
        "validation_episodes": len(validation_ids),
        "episode_overlap": sorted(train_ids & validation_ids),
        "rows": sum(item["row_count"] for item in per_episode.values()),
        "policy_rows": sum(item["policy_row_count"] for item in per_episode.values()),
        "value_rows": sum(item["value_row_count"] for item in per_episode.values()),
        "episodes": [per_episode[key] for key in sorted(per_episode)],
    }
    return manifest, train_ids, validation_ids


def collate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_options = max(len(row["options"]) for row in rows)
    max_actions = max(len(row["action"]) for row in rows)
    max_history = max(1, max(len(row.get("history") or []) for row in rows))
    batch = len(rows)
    states = torch.tensor([row["state"] for row in rows], dtype=torch.float32)
    options = torch.zeros((batch, max_options, ACTION_DIM), dtype=torch.float32)
    option_mask = torch.zeros((batch, max_options), dtype=torch.bool)
    actions = torch.zeros((batch, max_actions), dtype=torch.long)
    histories = torch.zeros((batch, max_history, HISTORY_DIM), dtype=torch.float32)
    history_mask = torch.zeros((batch, max_history), dtype=torch.bool)
    for index, row in enumerate(rows):
        count = len(row["options"])
        if count:
            options[index, :count] = torch.tensor(row["options"], dtype=torch.float32)
            option_mask[index, :count] = True
        if row["action"]:
            actions[index, :len(row["action"])] = torch.tensor(row["action"], dtype=torch.long)
        history = row.get("history") or []
        if history:
            histories[index, :len(history)] = torch.tensor(history, dtype=torch.float32)
            history_mask[index, :len(history)] = True
    result = {
        "states": states,
        "options": options,
        "option_mask": option_mask,
        "actions": actions,
        "action_lengths": torch.tensor([len(row["action"]) for row in rows], dtype=torch.long),
        "histories": histories,
        "history_lengths": torch.tensor([len(row.get("history") or []) for row in rows], dtype=torch.long),
        "history_mask": history_mask,
        "min_count": torch.tensor([row["min_count"] for row in rows], dtype=torch.long),
        "max_count": torch.tensor([row["max_count"] for row in rows], dtype=torch.long),
        "outcome": torch.tensor([row["outcome"] for row in rows], dtype=torch.float32),
        "policy_weight": torch.tensor([row["policy_weight"] for row in rows], dtype=torch.float32),
        "value_weight": torch.tensor([row["value_weight"] for row in rows], dtype=torch.float32),
        "rows": rows,
    }
    if all("option_card_ids" in row for row in rows):
        max_entities = max(1, max(len(row["entity_card_ids"]) for row in rows))
        max_deck = max(1, max(len(row.get("deck_card_ids") or []) for row in rows))
        option_card_ids = torch.zeros((batch, max_options), dtype=torch.long)
        option_target_card_ids = torch.zeros((batch, max_options), dtype=torch.long)
        option_attack_ids = torch.zeros((batch, max_options), dtype=torch.long)
        entity_card_ids = torch.zeros((batch, max_entities), dtype=torch.long)
        entity_zone_ids = torch.zeros((batch, max_entities), dtype=torch.long)
        entity_features = torch.zeros((batch, max_entities, ENTITY_DIM), dtype=torch.float32)
        entity_mask = torch.zeros((batch, max_entities), dtype=torch.bool)
        deck_card_ids = torch.zeros((batch, max_deck), dtype=torch.long)
        deck_mask = torch.zeros((batch, max_deck), dtype=torch.bool)
        for index, row in enumerate(rows):
            option_count = len(row["option_card_ids"])
            entity_count = len(row["entity_card_ids"])
            deck_count = len(row.get("deck_card_ids") or [])
            if option_count:
                option_card_ids[index, :option_count] = torch.tensor(row["option_card_ids"])
                option_target_card_ids[index, :option_count] = torch.tensor(row["option_target_card_ids"])
                option_attack_ids[index, :option_count] = torch.tensor(row["option_attack_ids"])
            if entity_count:
                entity_card_ids[index, :entity_count] = torch.tensor(row["entity_card_ids"])
                entity_zone_ids[index, :entity_count] = torch.tensor(row["entity_zone_ids"])
                entity_features[index, :entity_count] = torch.tensor(row["entity_features"])
                entity_mask[index, :entity_count] = True
            if deck_count:
                deck_card_ids[index, :deck_count] = torch.tensor(row["deck_card_ids"])
                deck_mask[index, :deck_count] = True
        result.update({
            "option_card_ids": option_card_ids,
            "option_target_card_ids": option_target_card_ids,
            "option_attack_ids": option_attack_ids,
            "entity_card_ids": entity_card_ids,
            "entity_zone_ids": entity_zone_ids,
            "entity_features": entity_features,
            "entity_mask": entity_mask,
            "deck_card_ids": deck_card_ids,
            "deck_mask": deck_mask,
        })
    return result


def make_loader(dataset: TrajectoryDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_rows, generator=torch.Generator().manual_seed(seed), num_workers=0)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def batch_loss(model: MaskedPointerActorCritic, batch: dict[str, Any], value_loss_weight: float = 0.25) -> tuple[torch.Tensor, dict[str, float]]:
    states, options = batch["states"], batch["options"]
    state, encoded_options, values = model.encode_batch(batch)
    batch_size, max_options = batch["option_mask"].shape
    selected = torch.zeros_like(batch["option_mask"])
    policy_sum = values.sum() * 0.0
    policy_count = values.sum() * 0.0
    max_steps = int(batch["action_lengths"].max().item()) + 1
    for step in range(max_steps):
        selected_count = torch.minimum(
            torch.full((batch_size,), step, dtype=torch.long, device=states.device),
            batch["action_lengths"],
        )
        logits = model.pointer_logits(state, encoded_options, selected, selected_count)
        legal = legal_choice_mask(batch["option_mask"], selected, selected_count, batch["min_count"], batch["max_count"])
        logits = logits.masked_fill(~legal, -torch.inf)
        active = step <= batch["action_lengths"]
        choosing = step < batch["action_lengths"]
        targets = torch.full((batch_size,), max_options, dtype=torch.long, device=states.device)
        if batch["actions"].shape[1] > step:
            targets = torch.where(choosing, batch["actions"][:, step], targets)
        if not bool(legal[active, targets[active]].all()):
            raise ValueError("teacher-forcing target is masked as illegal")
        per_row = F.cross_entropy(logits, targets, reduction="none")
        weights = batch["policy_weight"] * active.to(batch["policy_weight"].dtype)
        policy_sum = policy_sum + (per_row * weights).sum()
        policy_count = policy_count + weights.sum()
        if bool(choosing.any()):
            # PyTorch 1.10's CUDA boolean advanced-index assignment can hit an
            # internal shape assertion on mixed multi-select batches.  Scatter
            # expresses the same update and works across the heterogeneous
            # CUDA 11/12 training fleet.
            updates = torch.zeros_like(selected)
            safe_targets = targets.clamp_max(max_options - 1).unsqueeze(1)
            updates.scatter_(1, safe_targets, choosing.unsqueeze(1))
            selected = selected | updates
    policy_loss = policy_sum / policy_count.clamp_min(1.0)
    value_errors = F.mse_loss(values, batch["outcome"], reduction="none")
    value_sum = (value_errors * batch["value_weight"]).sum()
    value_count = batch["value_weight"].sum()
    value_loss = value_sum / value_count.clamp_min(1.0)
    loss = policy_loss + value_loss_weight * value_loss
    return loss, {
        "policy_sum": float(policy_sum.detach()), "policy_count": float(policy_count.detach()),
        "value_sum": float(value_sum.detach()), "value_count": float(value_count.detach()),
    }


@torch.inference_mode()
def greedy_decode(model: MaskedPointerActorCritic, batch: dict[str, Any]) -> list[list[int]]:
    outputs, _ = greedy_decode_with_confidence(model, batch)
    return outputs


@torch.inference_mode()
def greedy_decode_with_confidence(
    model: MaskedPointerActorCritic, batch: dict[str, Any]
) -> tuple[list[list[int]], list[float]]:
    state, encoded_options, _ = model.encode_batch(batch)
    batch_size, max_options = batch["option_mask"].shape
    selected = torch.zeros_like(batch["option_mask"])
    finished = torch.zeros(batch_size, dtype=torch.bool, device=state.device)
    outputs: list[list[int]] = [[] for _ in range(batch_size)]
    confidences: list[list[float]] = [[] for _ in range(batch_size)]
    max_steps = int(batch["max_count"].max().item()) + 1
    for step in range(max_steps):
        counts = selected.sum(dim=1)
        logits = model.pointer_logits(state, encoded_options, selected, counts)
        legal = legal_choice_mask(batch["option_mask"], selected, counts, batch["min_count"], batch["max_count"])
        logits = logits.masked_fill(~legal, -torch.inf)
        probabilities = torch.softmax(logits, dim=1)
        choices = logits.argmax(dim=1)
        for row_index, choice in enumerate(choices.tolist()):
            if finished[row_index]:
                continue
            confidences[row_index].append(float(probabilities[row_index, choice]))
            if choice == max_options:
                finished[row_index] = True
            else:
                outputs[row_index].append(choice)
                selected[row_index, choice] = True
        if bool(finished.all()):
            break
    if not bool(finished.all()):
        raise RuntimeError("decoder did not STOP within maxCount")
    return outputs, [min(values) if values else 0.0 for values in confidences]


def action_is_legal(action: list[int], option_count: int, min_count: int, max_count: int) -> bool:
    return min_count <= len(action) <= max_count and len(action) == len(set(action)) and all(0 <= index < option_count for index in action)


def run_epoch(
    model: MaskedPointerActorCritic,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int = 0,
    value_loss_weight: float = 0.25,
    gradient_clip_norm: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = Counter()
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            batch = _to_device(batch, device)
            loss, parts = batch_loss(model, batch, value_loss_weight=value_loss_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite RL-BC loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            totals.update(parts)
            totals["batches"] += 1
    if not totals["batches"]:
        raise ValueError("no batches processed")
    policy_loss = totals["policy_sum"] / max(1.0, totals["policy_count"])
    value_loss = totals["value_sum"] / max(1.0, totals["value_count"])
    return {"loss": policy_loss + value_loss_weight * value_loss, "policy_loss": policy_loss, "value_loss": value_loss, "policy_steps": int(totals["policy_count"]), "value_rows": int(totals["value_count"]), "batches": int(totals["batches"])}


@torch.inference_mode()
def evaluate_decoding(model: MaskedPointerActorCritic, loader: DataLoader, device: torch.device, max_batches: int = 0) -> dict[str, Any]:
    model.eval()
    totals = Counter()
    groups: dict[str, Counter] = defaultdict(Counter)
    for batch_index, original in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        batch = _to_device(original, device)
        predictions = greedy_decode(model, batch)
        for row, prediction in zip(original["rows"], predictions):
            target = row["action"]
            legal = action_is_legal(prediction, len(row["options"]), row["min_count"], row["max_count"])
            totals["decoded"] += 1
            totals["legal"] += int(legal)
            totals["invalid"] += int(not legal)
            if row["policy_weight"] != 1.0:
                continue
            totals["policy_rows"] += 1
            sequence_exact = prediction == target
            set_exact = set(prediction) == set(target)
            totals["sequence_exact"] += int(sequence_exact)
            totals["set_exact"] += int(set_exact)
            predicted_set, target_set = set(prediction), set(target)
            totals["candidate_tp"] += len(predicted_set & target_set)
            totals["candidate_fp"] += len(predicted_set - target_set)
            totals["candidate_fn"] += len(target_set - predicted_set)
            if len(target) == 0:
                totals["empty_rows"] += 1; totals["empty_correct"] += int(sequence_exact)
            if len(target) == 1 and row["max_count"] == 1:
                totals["single_rows"] += 1; totals["single_correct"] += int(sequence_exact)
            if len(target) > 1:
                totals["multi_rows"] += 1; totals["multi_correct"] += int(sequence_exact)
            for prefix, value in (("type", row["select_type"]), ("context", row["select_context"])):
                key = f"{prefix}:{value}"
                groups[key]["rows"] += 1
                groups[key]["sequence_exact"] += int(sequence_exact)
                groups[key]["set_exact"] += int(set_exact)
    precision_den = totals["candidate_tp"] + totals["candidate_fp"]
    recall_den = totals["candidate_tp"] + totals["candidate_fn"]
    def ratio(n: str, d: str) -> float | None:
        return totals[n] / totals[d] if totals[d] else None
    return {
        "policy_rows": totals["policy_rows"],
        "single_select_accuracy": ratio("single_correct", "single_rows"), "single_select_rows": totals["single_rows"],
        "sequence_exact_match": ratio("sequence_exact", "policy_rows"),
        "set_exact_match": ratio("set_exact", "policy_rows"),
        "candidate_precision": totals["candidate_tp"] / precision_den if precision_den else None,
        "candidate_recall": totals["candidate_tp"] / recall_den if recall_den else None,
        "empty_action_accuracy": ratio("empty_correct", "empty_rows"), "empty_action_rows": totals["empty_rows"],
        "multi_select_accuracy": ratio("multi_correct", "multi_rows"), "multi_select_rows": totals["multi_rows"],
        "decode_legal_rate": totals["legal"] / totals["decoded"] if totals["decoded"] else None,
        "invalid_actions": totals["invalid"], "decoded_rows": totals["decoded"],
        "by_select": {key: {"rows": value["rows"], "sequence_exact_match": value["sequence_exact"] / value["rows"], "set_exact_match": value["set_exact"] / value["rows"]} for key, value in sorted(groups.items())},
    }
