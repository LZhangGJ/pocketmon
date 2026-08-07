from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .bc import action_is_legal, collate_rows
from .features import (
    action_features,
    enhanced_context_features,
    state_features,
    structured_observation_features,
)
from .model import (
    MaskedPointerActorCritic,
    StructuredMaskedPointerActorCritic,
    StructuredTransformerMaskedPointerActorCritic,
    TemporalResourceBeliefTransformerActorCritic,
    legal_choice_mask,
)


STRUCTURED_ARCHITECTURE = "structured_card_attack_deepsets_deck_masked_pointer_with_stop"
STRUCTURED_TRANSFORMER_ARCHITECTURE = "structured_card_attack_transformer_text_deck_masked_pointer_with_stop"
V31_ARCHITECTURE = "structured_temporal_resource_belief_transformer_masked_pointer_with_stop"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path, device: torch.device) -> tuple[MaskedPointerActorCritic, dict[str, Any]]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config") or {}
    architecture = config.get("architecture")
    if architecture not in (STRUCTURED_ARCHITECTURE, STRUCTURED_TRANSFORMER_ARCHITECTURE, V31_ARCHITECTURE):
        raise ValueError(f"PPO requires a structured checkpoint, got {architecture!r}")
    if architecture == V31_ARCHITECTURE:
        model = TemporalResourceBeliefTransformerActorCritic(
            int(checkpoint["hidden_dim"]),
            history_length=max(1, int(config.get("history_length", 0))),
            use_history=bool(config.get("history_encoder", False)),
            use_resources=bool(config.get("v31_use_resources", False)),
            use_opponent_belief=bool(config.get("v31_use_opponent_belief", False)),
        )
        model.opponent_deck_prototypes = list(config.get("opponent_deck_prototypes") or [])
    else:
        model_class = (
            StructuredTransformerMaskedPointerActorCritic
            if architecture == STRUCTURED_TRANSFORMER_ARCHITECTURE else
            StructuredMaskedPointerActorCritic
        )
        model = model_class(int(checkpoint["hidden_dim"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return model, checkpoint


def model_row_from_observation(
    observation: dict[str, Any],
    deck: list[int],
    action: list[int] | None = None,
    *,
    history: list[list[float]] | None = None,
    enhanced_context: bool = False,
    opponent_prototypes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    select = observation.get("select")
    if not isinstance(select, dict):
        raise ValueError("PPO observation has no select object")
    options = select.get("option")
    if not isinstance(options, list):
        raise ValueError("PPO observation options must be a list")
    min_count = int(select.get("minCount", 0))
    max_count = int(select.get("maxCount", min_count))
    if min_count < 0 or max_count < min_count or max_count > len(options):
        raise ValueError("invalid PPO selection bounds")
    action = list(action or [])
    if action and not action_is_legal(action, len(options), min_count, max_count):
        raise ValueError("PPO action is not structurally legal")
    if len(deck) != 60 or not all(isinstance(card_id, int) and not isinstance(card_id, bool) for card_id in deck):
        raise ValueError("PPO deck must contain exactly 60 integer ids")
    row: dict[str, Any] = {
        "state": state_features(observation),
        "options": [
            action_features(option if isinstance(option, dict) else {}, index)
            for index, option in enumerate(options)
        ],
        "action": action,
        "history": list(history or []),
        "min_count": min_count,
        "max_count": max_count,
        "outcome": 0.0,
        "policy_weight": 1.0,
        "value_weight": 1.0,
    }
    row.update(structured_observation_features(observation, options))
    row["deck_card_ids"] = list(deck)
    if enhanced_context:
        row.update(enhanced_context_features(
            observation, options, deck, opponent_prototypes
        ))
    return row


def _runtime_context(model, history: list[list[float]] | None) -> dict[str, Any]:
    enhanced = isinstance(model, TemporalResourceBeliefTransformerActorCritic) and (
        model.use_resources or model.use_opponent_belief
    )
    return {
        "history": list(history or []) if isinstance(model, TemporalResourceBeliefTransformerActorCritic) else [],
        "enhanced_context": enhanced,
        "opponent_prototypes": list(getattr(model, "opponent_deck_prototypes", []) or []),
    }


def model_row_for_model(
    model,
    observation: dict[str, Any],
    deck: list[int],
    action: list[int] | None = None,
    history: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Create the exact row expected by a loaded legacy or V3.1 actor."""

    return model_row_from_observation(
        observation,
        deck,
        action,
        **_runtime_context(model, history),
    )


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def evaluate_action_sequences(
    model: MaskedPointerActorCritic,
    batch: dict[str, Any],
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return joint log-probability, mean token entropy and state value."""

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    state, encoded_options, values = model.encode_batch(batch)
    batch_size, max_options = batch["option_mask"].shape
    selected = torch.zeros_like(batch["option_mask"])
    log_probability = values * 0.0
    entropy_sum = values * 0.0
    token_count = values * 0.0
    max_steps = int(batch["action_lengths"].max().item()) + 1
    for step in range(max_steps):
        selected_count = torch.minimum(
            torch.full((batch_size,), step, dtype=torch.long, device=values.device),
            batch["action_lengths"],
        )
        logits = model.pointer_logits(state, encoded_options, selected, selected_count) / temperature
        legal = legal_choice_mask(
            batch["option_mask"], selected, selected_count, batch["min_count"], batch["max_count"]
        )
        # A finite floor preserves an effectively zero illegal probability
        # while avoiding 0 * -inf in entropy on heavily padded real batches.
        logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
        active = step <= batch["action_lengths"]
        choosing = step < batch["action_lengths"]
        targets = torch.full((batch_size,), max_options, dtype=torch.long, device=values.device)
        if batch["actions"].shape[1] > step:
            targets = torch.where(choosing, batch["actions"][:, step], targets)
        if not bool(legal[active, targets[active]].all()):
            raise ValueError("PPO action target is masked as illegal")
        log_probs = torch.log_softmax(logits, dim=1)
        probabilities = torch.softmax(logits, dim=1)
        chosen = log_probs.gather(1, targets[:, None]).squeeze(1)
        per_token_entropy = -torch.where(legal, probabilities * log_probs, torch.zeros_like(log_probs)).sum(dim=1)
        weights = active.to(values.dtype)
        log_probability = log_probability + torch.where(active, chosen, torch.zeros_like(chosen))
        entropy_sum = entropy_sum + per_token_entropy * weights
        token_count = token_count + weights
        if max_options and bool(choosing.any()):
            updates = torch.zeros_like(selected)
            updates.scatter_(1, targets.clamp_max(max_options - 1).unsqueeze(1), choosing.unsqueeze(1))
            selected = selected | updates
    return log_probability, entropy_sum / token_count.clamp_min(1.0), values


@torch.inference_mode()
def sample_action(
    model: MaskedPointerActorCritic,
    observation: dict[str, Any],
    deck: list[int],
    device: torch.device,
    temperature: float = 1.0,
    history: list[list[float]] | None = None,
) -> tuple[list[int], float, float, float]:
    """Sample one legal autoregressive action and record exact behavior statistics."""

    row = model_row_for_model(model, observation, deck, action=[], history=history)
    batch = to_device(collate_rows([row]), device)
    state, encoded_options, values = model.encode_batch(batch)
    max_options = batch["option_mask"].shape[1]
    selected = torch.zeros_like(batch["option_mask"])
    action: list[int] = []
    log_probability = 0.0
    entropy_sum = 0.0
    tokens = 0
    for _ in range(int(batch["max_count"][0].item()) + 1):
        counts = selected.sum(dim=1)
        logits = model.pointer_logits(state, encoded_options, selected, counts) / temperature
        legal = legal_choice_mask(
            batch["option_mask"], selected, counts, batch["min_count"], batch["max_count"]
        )
        logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(logits, dim=1)
        choice = int(torch.multinomial(probabilities[0], 1).item())
        log_probability += float(torch.log(probabilities[0, choice]).item())
        legal_probabilities = probabilities[0, legal[0]]
        entropy_sum += float((-(legal_probabilities * legal_probabilities.log())).sum().item())
        tokens += 1
        if choice == max_options:
            break
        action.append(choice)
        selected[0, choice] = True
    else:
        raise RuntimeError("sampled PPO action did not STOP within maxCount")
    if not action_is_legal(
        action,
        max_options,
        int(batch["min_count"][0].item()),
        int(batch["max_count"][0].item()),
    ):
        raise RuntimeError("sampled PPO action is illegal")
    return action, log_probability, float(values[0].item()), entropy_sum / max(1, tokens)


@torch.inference_mode()
def rank_single_actions(
    model: MaskedPointerActorCritic,
    observation: dict[str, Any],
    deck: list[int],
    device: torch.device,
    top_k: int,
    history: list[list[float]] | None = None,
) -> list[int]:
    """Rank legal option indices for a required single-select decision."""

    row = model_row_for_model(model, observation, deck, action=[], history=history)
    if row["min_count"] != 1 or row["max_count"] != 1:
        return []
    batch = to_device(collate_rows([row]), device)
    state, encoded_options, _ = model.encode_batch(batch)
    selected = torch.zeros_like(batch["option_mask"])
    counts = torch.zeros(1, dtype=torch.long, device=device)
    logits = model.pointer_logits(state, encoded_options, selected, counts)[0, :-1]
    logits = logits.masked_fill(~batch["option_mask"][0], torch.finfo(logits.dtype).min)
    count = min(max(1, int(top_k)), int(batch["option_mask"][0].sum().item()))
    return [int(index) for index in logits.topk(count).indices.tolist()]


@torch.inference_mode()
def predict_state_value(
    model: MaskedPointerActorCritic,
    observation: dict[str, Any],
    deck: list[int],
    device: torch.device,
    history: list[list[float]] | None = None,
) -> float:
    row = model_row_for_model(model, observation, deck, action=[], history=history)
    batch = to_device(collate_rows([row]), device)
    _, _, values = model.encode_batch(batch)
    return float(values[0].item())


def compute_gae(
    rows: list[dict[str, Any]],
    gamma: float = 0.997,
    gae_lambda: float = 0.95,
) -> list[dict[str, Any]]:
    if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    prepared = [dict(row) for row in rows]
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in prepared:
        groups[(str(row["episode_id"]), int(row["player"]))].append(row)
    for key, group in groups.items():
        group.sort(key=lambda row: int(row["action_step"]))
        if not group:
            continue
        if any(float(row.get("reward", 0.0)) != 0.0 for row in group[:-1]):
            raise ValueError(f"non-terminal reward before final decision for {key}")
        next_value = 0.0
        advantage = 0.0
        for row in reversed(group):
            old_value = float(row["behavior_value"])
            reward = float(row.get("reward", 0.0))
            delta = reward + gamma * next_value - old_value
            advantage = delta + gamma * gae_lambda * advantage
            row["advantage"] = advantage
            row["return"] = advantage + old_value
            next_value = old_value
    return prepared


def normalize_advantages(rows: list[dict[str, Any]], epsilon: float = 1e-8) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("cannot normalize empty PPO rows")
    values = torch.tensor([float(row["advantage"]) for row in rows], dtype=torch.float64)
    mean = float(values.mean())
    std = float(values.std(unbiased=False))
    prepared = [dict(row) for row in rows]
    for row in prepared:
        row["advantage"] = (float(row["advantage"]) - mean) / max(std, epsilon)
    return prepared


def ppo_batch_loss(
    model: MaskedPointerActorCritic,
    batch: dict[str, Any],
    clip_ratio: float = 0.1,
    value_clip: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    new_log_probability, entropy, values = evaluate_action_sequences(model, batch)
    rows = batch["rows"]
    old_log_probability = torch.tensor(
        [float(row["behavior_log_probability"]) for row in rows], dtype=values.dtype, device=values.device
    )
    old_values = torch.tensor(
        [float(row["behavior_value"]) for row in rows], dtype=values.dtype, device=values.device
    )
    advantages = torch.tensor(
        [float(row["advantage"]) for row in rows], dtype=values.dtype, device=values.device
    )
    returns = torch.tensor(
        [float(row["return"]) for row in rows], dtype=values.dtype, device=values.device
    )
    # PPO clipping makes ratios far outside this interval equivalent for the
    # clipped surrogate; bounding the exponent prevents overflow before the
    # min/clamp operation can take effect.
    log_ratio = (new_log_probability - old_log_probability).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    clipped_values = old_values + (values - old_values).clamp(-value_clip, value_clip)
    value_loss = 0.5 * torch.maximum((values - returns).square(), (clipped_values - returns).square()).mean()
    entropy_mean = entropy.mean()
    loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy_mean
    with torch.no_grad():
        approximate_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_ratio).to(values.dtype).mean()
    return loss, {
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy_mean.detach()),
        "approximate_kl": float(approximate_kl.detach()),
        "clip_fraction": float(clip_fraction.detach()),
        "ratio_mean": float(ratio.mean().detach()),
        "value_mean": float(values.mean().detach()),
    }
