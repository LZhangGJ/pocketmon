from __future__ import annotations

import copy
import importlib.util
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import Experiment7Error, sha256_file


ARCHITECTURE = "experiment7_universal_deck8_autoregressive_stop"
ROLLOUT_FORMAT = "experiment7_universal_ppo_v1"


@dataclass(frozen=True)
class FeatureRuntime:
    features: Any
    tokenizer: Any
    cards: dict[int, dict[str, Any]]
    attacks: dict[int, dict[str, Any]]
    entity_cards: dict[int, dict[str, Any]]


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_feature_runtime(reference_root: Path, engine_catalog: Path) -> FeatureRuntime:
    data_pipeline = reference_root.resolve() / "data_pipeline"
    features = _load_module("experiment7_ppo_features", data_pipeline / "features.py")
    tokenizer = _load_module("experiment7_ppo_tokenizer", data_pipeline / "tokenizer.py")
    cards, attacks = features.load_catalog(engine_catalog.resolve())
    return FeatureRuntime(
        features=features,
        tokenizer=tokenizer,
        cards=cards,
        attacks=attacks,
        entity_cards=tokenizer.load_cards(engine_catalog.resolve()),
    )


def load_universal_checkpoint(
    checkpoint: Path, reference_root: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    training = reference_root.resolve() / "training"
    integration = Path(__file__).resolve().parent
    for path in (training, integration):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from universal_deck_model import UniversalDeckModelConfig, UniversalDeckTransformerPolicy

    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    if payload.get("architecture") != ARCHITECTURE:
        raise Experiment7Error(
            f"Universal PPO requires {ARCHITECTURE}, got {payload.get('architecture')!r}"
        )
    config = UniversalDeckModelConfig(**payload["config"])
    model = UniversalDeckTransformerPolicy(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    return model, payload


def sanitize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    sanitized = copy.deepcopy(observation)
    current = sanitized.get("current")
    if not isinstance(current, dict):
        return sanitized
    players = current.get("players")
    if not isinstance(players, list):
        return sanitized
    actor = int(current.get("yourIndex", 0) or 0)
    for index, player in enumerate(players):
        if not isinstance(player, dict):
            continue
        prize = player.get("prize")
        if isinstance(prize, list):
            player["prize"] = [{} for _ in prize]
        if index != actor:
            player["hand"] = []
    return sanitized


def live_row(
    observation: dict[str, Any],
    deck: list[int],
    history: list[tuple[np.ndarray, np.ndarray]],
    runtime: FeatureRuntime,
    config: Any,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    select = observation.get("select")
    if not isinstance(select, dict) or not isinstance(select.get("option"), list):
        raise Experiment7Error("Universal PPO observation has no selection options")
    if len(deck) != int(config.deck_size):
        raise Experiment7Error(f"Universal PPO deck must have {config.deck_size} cards")
    options = select["option"]
    option_count = len(options)
    minimum = max(0, min(option_count, int(select.get("minCount", 0) or 0)))
    maximum = max(minimum, min(option_count, int(select.get("maxCount", 0) or 0)))
    visible = sanitize_observation(observation)
    state = np.asarray(runtime.features.encode_state(visible), dtype=np.float32)
    option_rows = np.asarray(
        [
            runtime.features.encode_option(visible, option, index, runtime.cards, runtime.attacks)
            for index, option in enumerate(options)
        ],
        dtype=np.float32,
    )
    if option_count == 0:
        option_rows = np.zeros((0, int(config.option_dim)), dtype=np.float32)
    entity_cat, entity_num, entity_mask, truncated = runtime.tokenizer.encode_entities(
        visible, runtime.entity_cards
    )
    history_state = np.zeros(
        (int(config.history_length), int(config.state_dim)), dtype=np.float32
    )
    history_action = np.zeros(
        (int(config.history_length), int(config.option_dim)), dtype=np.float32
    )
    history_mask = np.zeros(int(config.history_length), dtype=np.uint8)
    retained = history[-int(config.history_length) :]
    offset = int(config.history_length) - len(retained)
    for slot, (past_state, past_action) in enumerate(retained, start=offset):
        history_state[slot] = past_state
        history_action[slot] = past_action
        history_mask[slot] = 1
    row = {
        "state": state.tolist(),
        "history_state": history_state.tolist(),
        "history_action": history_action.tolist(),
        "history_mask": history_mask.tolist(),
        "own_deck_cards": list(deck),
        "entity_cat": np.asarray(entity_cat, dtype=np.int64).tolist(),
        "entity_num": np.asarray(entity_num, dtype=np.float32).tolist(),
        "entity_mask": np.asarray(entity_mask, dtype=np.uint8).tolist(),
        "options": option_rows.tolist(),
        "min_count": minimum,
        "max_count": maximum,
        "truncated_entities": int(truncated),
    }
    return row, state, option_rows


def append_history(
    history: list[tuple[np.ndarray, np.ndarray]],
    state: np.ndarray,
    option_rows: np.ndarray,
    action: list[int],
    history_length: int,
) -> None:
    action_row = np.zeros(option_rows.shape[1] if option_rows.ndim == 2 else 176, dtype=np.float32)
    if action:
        action_row = option_rows[action].mean(axis=0).astype(np.float32)
    history.append((state.astype(np.float32, copy=True), action_row))
    del history[:-history_length]


def collate_rows(rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot collate empty Universal PPO rows")
    batch = len(rows)
    max_entities = max(len(row["entity_mask"]) for row in rows)
    max_options = max(len(row["options"]) for row in rows)
    state_dim = len(rows[0]["state"])
    history_length = len(rows[0]["history_mask"])
    option_dim = len(rows[0]["history_action"][0])
    entity_cat_dim = len(rows[0]["entity_cat"][0])
    entity_num_dim = len(rows[0]["entity_num"][0])
    state = np.zeros((batch, state_dim), dtype=np.float32)
    history_state = np.zeros((batch, history_length, state_dim), dtype=np.float32)
    history_action = np.zeros((batch, history_length, option_dim), dtype=np.float32)
    history_mask = np.zeros((batch, history_length), dtype=np.uint8)
    own_deck_cards = np.zeros((batch, 60), dtype=np.int64)
    entity_cat = np.zeros((batch, max_entities, entity_cat_dim), dtype=np.int64)
    entity_num = np.zeros((batch, max_entities, entity_num_dim), dtype=np.float32)
    entity_mask = np.zeros((batch, max_entities), dtype=np.uint8)
    options = np.zeros((batch, max_options, option_dim), dtype=np.float32)
    option_mask = np.zeros((batch, max_options), dtype=np.uint8)
    for index, row in enumerate(rows):
        state[index] = np.asarray(row["state"], dtype=np.float32)
        history_state[index] = np.asarray(row["history_state"], dtype=np.float32)
        history_action[index] = np.asarray(row["history_action"], dtype=np.float32)
        history_mask[index] = np.asarray(row["history_mask"], dtype=np.uint8)
        own_deck_cards[index] = np.asarray(row["own_deck_cards"], dtype=np.int64)
        entities = len(row["entity_mask"])
        entity_cat[index, :entities] = np.asarray(row["entity_cat"], dtype=np.int64)
        entity_num[index, :entities] = np.asarray(row["entity_num"], dtype=np.float32)
        entity_mask[index, :entities] = np.asarray(row["entity_mask"], dtype=np.uint8)
        count = len(row["options"])
        if count:
            options[index, :count] = np.asarray(row["options"], dtype=np.float32)
            option_mask[index, :count] = 1
    tensor = lambda value: torch.from_numpy(value).to(device)
    return {
        "state": tensor(state),
        "history_state": tensor(history_state),
        "history_action": tensor(history_action),
        "history_mask": tensor(history_mask),
        "own_deck_cards": tensor(own_deck_cards),
        "entity_cat": tensor(entity_cat),
        "entity_num": tensor(entity_num),
        "entity_mask": tensor(entity_mask),
        "options": tensor(options),
        "option_mask": tensor(option_mask),
        "min_count": torch.tensor([int(row["min_count"]) for row in rows], device=device),
        "max_count": torch.tensor([int(row["max_count"]) for row in rows], device=device),
        "rows": rows,
    }


def forward_model(model: torch.nn.Module, batch: dict[str, Any]) -> Any:
    return model(
        batch["state"],
        batch["history_state"],
        batch["history_action"],
        batch["history_mask"],
        batch["own_deck_cards"],
        batch["entity_cat"],
        batch["entity_num"],
        batch["entity_mask"],
        batch["options"],
        batch["option_mask"],
    )


def evaluate_actions(
    model: torch.nn.Module, batch: dict[str, Any], temperature: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    encoding = forward_model(model, batch)
    rows = batch["rows"]
    lengths = torch.tensor([len(row["action"]) for row in rows], device=encoding.global_hidden.device)
    selected = torch.zeros_like(batch["option_mask"], dtype=torch.bool)
    log_probability = encoding.value_logits * 0.0
    entropy_sum = encoding.value_logits * 0.0
    token_count = encoding.value_logits * 0.0
    option_count = batch["option_mask"].shape[1]
    for step in range(int(lengths.max().item()) + 1):
        active = step <= lengths
        choosing = step < lengths
        logits = model.decoder_logits(
            encoding, selected, batch["min_count"], batch["max_count"]
        ) / temperature
        selected_count = selected.sum(dim=1)
        legal_options = (
            batch["option_mask"].bool()
            & ~selected
            & (selected_count < batch["max_count"])[:, None]
        )
        legal = torch.cat(
            (legal_options, (selected_count >= batch["min_count"])[:, None]), dim=1
        )
        targets = torch.full((len(rows),), option_count, dtype=torch.long, device=logits.device)
        for index, row in enumerate(rows):
            if step < len(row["action"]):
                targets[index] = int(row["action"][step])
        if not bool(legal[active, targets[active]].all()):
            raise Experiment7Error("Universal PPO action target is masked as illegal")
        logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
        log_probs = torch.log_softmax(logits, dim=1)
        probabilities = torch.softmax(logits, dim=1)
        chosen = log_probs.gather(1, targets[:, None]).squeeze(1)
        entropy = -torch.where(
            legal, probabilities * log_probs, torch.zeros_like(log_probs)
        ).sum(dim=1)
        log_probability = log_probability + torch.where(active, chosen, torch.zeros_like(chosen))
        weights = active.to(entropy.dtype)
        entropy_sum = entropy_sum + entropy * weights
        token_count = token_count + weights
        next_selected = selected.clone()
        for index, row in enumerate(rows):
            if step < len(row["action"]):
                next_selected[index, int(row["action"][step])] = True
        selected = next_selected
    values = torch.tanh(encoding.value_logits / 2.0)
    return log_probability, entropy_sum / token_count.clamp_min(1.0), values


@torch.inference_mode()
def sample_action(
    model: torch.nn.Module,
    row: dict[str, Any],
    device: torch.device,
    temperature: float = 1.0,
) -> tuple[list[int], float, float, float]:
    batch = collate_rows([{**row, "action": []}], device)
    encoding = forward_model(model, batch)
    selected = torch.zeros_like(batch["option_mask"], dtype=torch.bool)
    option_count = selected.shape[1]
    action: list[int] = []
    log_probability = 0.0
    entropy_sum = 0.0
    for tokens in range(int(batch["max_count"][0].item()) + 1):
        logits = model.decoder_logits(
            encoding, selected, batch["min_count"], batch["max_count"]
        )[0] / temperature
        probabilities = torch.softmax(logits.float(), dim=0)
        choice = int(torch.multinomial(probabilities, 1).item())
        log_probability += float(torch.log(probabilities[choice]).item())
        legal_probabilities = probabilities[probabilities > 0]
        entropy_sum += float((-(legal_probabilities * legal_probabilities.log())).sum().item())
        if choice == option_count:
            break
        action.append(choice)
        selected[0, choice] = True
    else:
        raise Experiment7Error("Universal PPO sampled action did not STOP")
    minimum, maximum = int(row["min_count"]), int(row["max_count"])
    if not (
        minimum <= len(action) <= maximum
        and len(action) == len(set(action))
        and all(0 <= index < option_count for index in action)
    ):
        raise Experiment7Error("Universal PPO sampled an illegal action")
    value = float(torch.tanh(encoding.value_logits[0] / 2.0).item())
    return action, log_probability, value, entropy_sum / max(tokens + 1, 1)


def compute_gae(
    rows: list[dict[str, Any]], gamma: float = 0.997, gae_lambda: float = 0.95
) -> list[dict[str, Any]]:
    prepared = [dict(row) for row in rows]
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in prepared:
        groups[(str(row["episode_id"]), int(row["player"]))].append(row)
    for key, group in groups.items():
        group.sort(key=lambda row: int(row["action_step"]))
        if any(float(row.get("reward", 0.0)) != 0.0 for row in group[:-1]):
            raise Experiment7Error(f"non-terminal reward before final decision for {key}")
        next_value = 0.0
        advantage = 0.0
        for row in reversed(group):
            value = float(row["behavior_value"])
            delta = float(row.get("reward", 0.0)) + gamma * next_value - value
            advantage = delta + gamma * gae_lambda * advantage
            row["advantage"] = advantage
            row["return"] = advantage + value
            next_value = value
    return prepared


def normalize_advantages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = np.asarray([float(row["advantage"]) for row in rows], dtype=np.float64)
    if not len(values):
        raise ValueError("cannot normalize empty Universal PPO rows")
    scale = max(float(values.std()), 1e-8)
    return [{**row, "advantage": (float(row["advantage"]) - float(values.mean())) / scale} for row in rows]


def ppo_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    clip_ratio: float = 0.1,
    value_clip: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    teacher_anchor_coefficient: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    # PPO ratios must compare the exact behavior distribution.  Keep dropout
    # disabled while retaining autograd; otherwise identical weights can start
    # an update with ratio != 1 solely because of a new dropout mask.
    was_training = model.training
    model.eval()
    new_log_probability, entropy, values = evaluate_actions(model, batch)
    if was_training:
        model.train()
    rows = batch["rows"]
    tensor = lambda key: torch.tensor(
        [float(row[key]) for row in rows], dtype=values.dtype, device=values.device
    )
    old_log_probability = tensor("behavior_log_probability")
    old_values = tensor("behavior_value")
    advantages = tensor("advantage")
    returns = tensor("return")
    log_ratio = (new_log_probability - old_log_probability).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    policy = -torch.minimum(
        ratio * advantages,
        ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages,
    ).mean()
    clipped_values = old_values + (values - old_values).clamp(-value_clip, value_clip)
    value = 0.5 * torch.maximum(
        (values - returns).square(), (clipped_values - returns).square()
    ).mean()
    teacher = torch.tensor(
        [float(row.get("teacher_log_probability", row["behavior_log_probability"])) for row in rows],
        dtype=values.dtype,
        device=values.device,
    )
    teacher_anchor = (new_log_probability - teacher).square().mean()
    entropy_mean = entropy.mean()
    loss = (
        policy
        + value_coefficient * value
        + teacher_anchor_coefficient * teacher_anchor
        - entropy_coefficient * entropy_mean
    )
    with torch.no_grad():
        approximate_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_ratio).to(values.dtype).mean()
    return loss, {
        "policyLoss": float(policy.detach()),
        "valueLoss": float(value.detach()),
        "entropy": float(entropy_mean.detach()),
        "teacherAnchor": float(teacher_anchor.detach()),
        "approximateKl": float(approximate_kl.detach()),
        "clipFraction": float(clip_fraction.detach()),
        "ratioMean": float(ratio.mean().detach()),
    }


def checkpoint_sha256(path: Path) -> str:
    return sha256_file(path)
