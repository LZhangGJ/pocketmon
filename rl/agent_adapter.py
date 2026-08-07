from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Callable

import torch

from .bc import action_is_legal, collate_rows, greedy_decode_with_confidence
from .action_q import ActionValueEnsemble, q_mean_and_std
from .features import action_features, history_features, state_features, structured_observation_features
from .model import (
    MaskedPointerActorCritic,
    StructuredMaskedPointerActorCritic,
    StructuredTransformerMaskedPointerActorCritic,
)


STATELESS_ARCHITECTURE = "stateless_masked_autoregressive_candidate_pointer_with_stop"
HISTORY_ARCHITECTURE = "causal_gru_history_masked_autoregressive_candidate_pointer_with_stop"
STRUCTURED_ARCHITECTURE = "structured_card_attack_deepsets_deck_masked_pointer_with_stop"
STRUCTURED_TRANSFORMER_ARCHITECTURE = "structured_card_attack_transformer_text_deck_masked_pointer_with_stop"


def conservative_q_choice(
    actor_choice: int,
    candidates: torch.Tensor,
    lower_bounds: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    min_margin: float,
    max_uncertainty: float,
) -> tuple[int, str, float, float]:
    """Let Q override the actor only with a calibrated, low-variance margin."""
    best_position = int(lower_bounds.argmax().item())
    best_choice = int(candidates[best_position].item())
    actor_positions = (candidates == actor_choice).nonzero(as_tuple=False)
    if not len(actor_positions):
        raise ValueError("actor choice must be present in Q candidates")
    actor_position = int(actor_positions[0].item())
    margin = float((lower_bounds[best_position] - lower_bounds[actor_position]).item())
    best_uncertainty = float(uncertainty[best_position].item())
    if best_choice == actor_choice:
        return actor_choice, "agreement", margin, best_uncertainty
    if margin >= min_margin and best_uncertainty <= max_uncertainty:
        return best_choice, "override", margin, best_uncertainty
    return actor_choice, "abstain", margin, best_uncertainty


def apply_q_override_budget(
    actor_choice: int,
    q_choice: int,
    q_status: str,
    credit: float,
    max_override_rate: float,
) -> tuple[int, str, float]:
    """Apply a token-bucket cap so Q cannot gradually take over the actor."""

    updated_credit = min(1.0, float(credit) + float(max_override_rate))
    if q_status != "override":
        return q_choice, q_status, updated_credit
    if updated_credit + 1e-12 < 1.0:
        return actor_choice, "budget_abstain", updated_credit
    return q_choice, q_status, updated_credit - 1.0


class RLBCPolicyAdapter:
    """Local-game adapter with masked decode and a validated rule fallback."""

    def __init__(
        self,
        checkpoint_path: Path,
        fallback: Callable[[dict[str, Any]], list[int]],
        device: str = "cpu",
        confidence_threshold: float | None = None,
        deck: list[int] | None = None,
        q_checkpoint_path: Path | None = None,
        q_top_k: int = 4,
        q_uncertainty_penalty: float = 0.25,
        q_min_margin: float | None = None,
        q_max_uncertainty: float | None = None,
        q_max_override_rate: float = 1.0,
        q_min_validation_rows: int = 0,
        q_max_validation_mae: float | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.fallback = fallback
        self.device = torch.device(device)
        self._model: MaskedPointerActorCritic | None = None
        self._q_checkpoint_path = Path(q_checkpoint_path) if q_checkpoint_path else None
        self._q_model: ActionValueEnsemble | None = None
        self._q_top_k = max(1, int(q_top_k))
        self._q_uncertainty_penalty = float(q_uncertainty_penalty)
        self._q_auto_margin = q_min_margin is None
        self._q_auto_uncertainty = q_max_uncertainty is None
        self._q_min_margin = 0.15 if q_min_margin is None else float(q_min_margin)
        self._q_max_uncertainty = 0.15 if q_max_uncertainty is None else float(q_max_uncertainty)
        self._q_max_override_rate = float(q_max_override_rate)
        self._q_min_validation_rows = int(q_min_validation_rows)
        self._q_max_validation_mae = (
            math.inf if q_max_validation_mae is None else float(q_max_validation_mae)
        )
        self._q_override_credit = 0.0
        if not math.isfinite(self._q_min_margin) or self._q_min_margin < 0:
            raise ValueError("Q minimum margin must be finite and non-negative")
        if not math.isfinite(self._q_max_uncertainty) or self._q_max_uncertainty < 0:
            raise ValueError("Q maximum uncertainty must be finite and non-negative")
        if not math.isfinite(self._q_max_override_rate) or not 0.0 <= self._q_max_override_rate <= 1.0:
            raise ValueError("Q maximum override rate must be finite and in [0, 1]")
        if self._q_min_validation_rows < 0:
            raise ValueError("Q minimum validation rows must be non-negative")
        if self._q_max_validation_mae < 0 or math.isnan(self._q_max_validation_mae):
            raise ValueError("Q maximum validation MAE must be non-negative")
        self._last_q_margin: float | None = None
        self._last_q_uncertainty: float | None = None
        self._structured = False
        if deck is not None and (
            len(deck) != 60
            or not all(isinstance(card_id, int) and not isinstance(card_id, bool) for card_id in deck)
        ):
            raise ValueError("configured deck must contain exactly 60 integer card ids")
        self._configured_deck = list(deck or [])
        self._deck: list[int] = list(self._configured_deck)
        self._confidence_threshold_override = confidence_threshold
        self._confidence_threshold = 0.0
        self._last_confidence: float | None = None
        self._history_enabled = False
        self._history_length = 0
        self._history: list[list[float]] = []
        self._load_attempted = False
        self._last_turn: int | None = None
        self._diagnostics = {
            "model_actions": 0,
            "fallback_actions": 0,
            "load_errors": 0,
            "inference_errors": 0,
            "illegal_model_actions": 0,
            "low_confidence_actions": 0,
            "illegal_fallback_actions": 0,
            "emergency_legal_actions": 0,
            "q_load_errors": 0,
            "q_actions": 0,
            "q_agreements": 0,
            "q_overrides": 0,
            "q_abstentions": 0,
            "q_budget_abstentions": 0,
        }

    def reset(self) -> None:
        self._history.clear()
        self._deck = list(self._configured_deck)
        self._last_turn = None

    def _load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            except TypeError:
                # torch<2.0 does not expose the weights_only keyword. These are
                # trusted, locally trained checkpoints rather than user uploads.
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            config = checkpoint["config"]
            architecture = config.get("architecture", STATELESS_ARCHITECTURE)
            if architecture not in (
                STATELESS_ARCHITECTURE,
                HISTORY_ARCHITECTURE,
                STRUCTURED_ARCHITECTURE,
                STRUCTURED_TRANSFORMER_ARCHITECTURE,
            ):
                raise ValueError(f"unsupported checkpoint architecture: {architecture}")
            self._history_enabled = architecture == HISTORY_ARCHITECTURE
            self._structured = architecture in (STRUCTURED_ARCHITECTURE, STRUCTURED_TRANSFORMER_ARCHITECTURE)
            self._history_length = int(config.get("history_length", 0)) if self._history_enabled else 0
            if self._history_enabled and self._history_length <= 0:
                raise ValueError("history checkpoint has no positive history_length")
            configured_threshold = float(config.get("confidence_threshold", 0.0))
            self._confidence_threshold = (
                configured_threshold if self._confidence_threshold_override is None
                else float(self._confidence_threshold_override)
            )
            if not 0.0 <= self._confidence_threshold <= 1.0:
                raise ValueError("confidence threshold must be in [0, 1]")
            if architecture == STRUCTURED_TRANSFORMER_ARCHITECTURE:
                model = StructuredTransformerMaskedPointerActorCritic(int(checkpoint["hidden_dim"]))
            elif architecture == STRUCTURED_ARCHITECTURE:
                model = StructuredMaskedPointerActorCritic(int(checkpoint["hidden_dim"]))
            else:
                model = MaskedPointerActorCritic(
                    int(checkpoint["hidden_dim"]), history_encoder=self._history_enabled
                )
            model = model.to(self.device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            self._model = model
            if self._q_checkpoint_path and self._q_checkpoint_path.is_file() and self._structured:
                try:
                    try:
                        q_checkpoint = torch.load(self._q_checkpoint_path, map_location=self.device, weights_only=False)
                    except TypeError:
                        q_checkpoint = torch.load(self._q_checkpoint_path, map_location=self.device)
                    actor_sha = hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest()
                    if q_checkpoint.get("actor_checkpoint_sha256") != actor_sha:
                        raise ValueError("action-Q checkpoint was trained for a different actor")
                    q_model = ActionValueEnsemble(
                        int(q_checkpoint["hidden_dim"]), int(q_checkpoint["heads"])
                    ).to(self.device)
                    q_model.load_state_dict(q_checkpoint["model"])
                    q_model.eval()
                    self._q_model = q_model
                    validation = q_checkpoint.get("validation", {})
                    validation_rows = int(validation.get("rows", 0))
                    validation_mae = float(validation.get("mae", math.inf))
                    if validation_rows < self._q_min_validation_rows:
                        raise ValueError(
                            f"action-Q validation rows {validation_rows} < {self._q_min_validation_rows}"
                        )
                    if not math.isfinite(validation_mae) or validation_mae > self._q_max_validation_mae:
                        raise ValueError(
                            f"action-Q validation MAE {validation_mae} > {self._q_max_validation_mae}"
                        )
                    if self._q_auto_margin:
                        self._q_min_margin = max(0.10, min(0.50, 0.50 * validation_mae))
                    if self._q_auto_uncertainty:
                        mean_uncertainty = float(validation.get("mean_uncertainty", 0.10))
                        self._q_max_uncertainty = max(0.05, min(0.25, 1.50 * mean_uncertainty))
                except Exception:
                    self._diagnostics["q_load_errors"] += 1
                    self._q_model = None
        except Exception:
            self._diagnostics["load_errors"] += 1
            self._model = None

    @staticmethod
    def _selection(observation: dict[str, Any]) -> tuple[list[Any], int, int] | None:
        select = observation.get("select")
        if not isinstance(select, dict):
            return None
        options = select.get("option")
        if not isinstance(options, list):
            return None
        min_count = int(select.get("minCount", 0))
        max_count = int(select.get("maxCount", min_count))
        if min_count < 0 or max_count < min_count or max_count > len(options):
            return None
        return options, min_count, max_count

    @staticmethod
    def _legal(action: Any, option_count: int, min_count: int, max_count: int) -> bool:
        return isinstance(action, list) and all(
            isinstance(index, int) and not isinstance(index, bool) for index in action
        ) and action_is_legal(action, option_count, min_count, max_count)

    @staticmethod
    def _emergency_action(option_count: int, min_count: int) -> list[int]:
        return list(range(min(min_count, option_count)))

    def _model_action(
        self,
        observation: dict[str, Any],
        options: list[Any],
        min_count: int,
        max_count: int,
    ) -> tuple[list[int], list[float], list[list[float]], float]:
        if self._model is None:
            raise RuntimeError("model unavailable")
        state = state_features(observation)
        encoded_options = [
            action_features(option if isinstance(option, dict) else {}, index)
            for index, option in enumerate(options)
        ]
        row = {
            "state": state,
            "options": encoded_options,
            "action": [],
            "history": self._history[-self._history_length:] if self._history_enabled else [],
            "min_count": min_count,
            "max_count": max_count,
            "outcome": 0.0,
            "policy_weight": 0.0,
            "value_weight": 1.0,
        }
        if self._structured:
            row.update(structured_observation_features(observation, options))
            row["deck_card_ids"] = list(self._deck)
        batch = collate_rows([row])
        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        if self._q_model is not None and min_count == 1 and max_count == 1 and len(options) > 1:
            with torch.inference_mode():
                actor_state, actor_options, _ = self._model.encode_batch(batch)
                selected = torch.zeros_like(batch["option_mask"])
                counts = torch.zeros(1, dtype=torch.long, device=self.device)
                logits = self._model.pointer_logits(actor_state, actor_options, selected, counts)[0, :-1]
                logits = logits.masked_fill(~batch["option_mask"][0], torch.finfo(logits.dtype).min)
                count = min(self._q_top_k, len(options))
                candidates = logits.topk(count).indices
                q_values = self._q_model(actor_state, actor_options)
                q_mean, q_std = q_mean_and_std(q_values)
                lower = q_mean[0, candidates] - self._q_uncertainty_penalty * q_std[0, candidates]
                actor_choice = int(logits.argmax().item())
                chosen, q_status, margin, best_uncertainty = conservative_q_choice(
                    actor_choice,
                    candidates,
                    lower,
                    q_std[0, candidates],
                    min_margin=self._q_min_margin,
                    max_uncertainty=self._q_max_uncertainty,
                )
                chosen, q_status, self._q_override_credit = apply_q_override_budget(
                    actor_choice,
                    chosen,
                    q_status,
                    self._q_override_credit,
                    self._q_max_override_rate,
                )
                confidence = float(torch.softmax(logits, dim=0)[chosen].item())
                self._diagnostics["q_actions"] += 1
                diagnostic_key = {
                    "agreement": "q_agreements",
                    "override": "q_overrides",
                    "abstain": "q_abstentions",
                    "budget_abstain": "q_budget_abstentions",
                }[q_status]
                self._diagnostics[diagnostic_key] += 1
                self._last_q_margin = margin
                self._last_q_uncertainty = best_uncertainty
                return [chosen], state, encoded_options, confidence
        actions, confidences = greedy_decode_with_confidence(self._model, batch)
        return actions[0], state, encoded_options, confidences[0]

    def _remember(
        self,
        state: list[float],
        encoded_options: list[list[float]],
        action: list[int],
    ) -> None:
        if self._history_enabled:
            self._history.append(history_features(state, encoded_options, action))
            self._history = self._history[-self._history_length:]

    def act(self, observation: dict[str, Any]) -> list[int]:
        selection = self._selection(observation)
        if selection is None:
            self.reset()
            action = self.fallback(observation)
            if (
                isinstance(action, list) and len(action) == 60
                and all(isinstance(card_id, int) and not isinstance(card_id, bool) for card_id in action)
            ):
                self._deck = list(action)
            return action
        options, min_count, max_count = selection
        current = observation.get("current") or {}
        turn = current.get("turn")
        if isinstance(turn, int) and self._last_turn is not None and turn < self._last_turn:
            self.reset()
        if isinstance(turn, int):
            self._last_turn = turn

        self._load()
        state = state_features(observation)
        encoded_options = [
            action_features(option if isinstance(option, dict) else {}, index)
            for index, option in enumerate(options)
        ]
        try:
            action, state, encoded_options, confidence = self._model_action(
                observation, options, min_count, max_count
            )
            self._last_confidence = confidence
            if not self._legal(action, len(options), min_count, max_count):
                self._diagnostics["illegal_model_actions"] += 1
                raise ValueError("model returned an illegal action")
            if confidence < self._confidence_threshold:
                self._diagnostics["low_confidence_actions"] += 1
                action = self.fallback(observation)
                self._diagnostics["fallback_actions"] += 1
                if not self._legal(action, len(options), min_count, max_count):
                    self._diagnostics["illegal_fallback_actions"] += 1
                    action = self._emergency_action(len(options), min_count)
                    self._diagnostics["emergency_legal_actions"] += 1
            else:
                self._diagnostics["model_actions"] += 1
        except Exception:
            self._diagnostics["inference_errors"] += 1
            action = self.fallback(observation)
            self._diagnostics["fallback_actions"] += 1
            if not self._legal(action, len(options), min_count, max_count):
                self._diagnostics["illegal_fallback_actions"] += 1
                action = self._emergency_action(len(options), min_count)
                self._diagnostics["emergency_legal_actions"] += 1
        if not self._legal(action, len(options), min_count, max_count):
            raise RuntimeError("adapter failed to produce a structurally legal action")
        self._remember(state, encoded_options, action)
        return action

    def diagnostics(self) -> dict[str, Any]:
        result = dict(self._diagnostics)
        result.update({
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_exists": self.checkpoint_path.is_file(),
            "checkpoint_sha256": (
                hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest()
                if self.checkpoint_path.is_file() else None
            ),
            "history_enabled": self._history_enabled,
            "history_tokens": len(self._history),
            "structured": self._structured,
            "deck_cards": len(self._deck),
            "confidence_threshold": self._confidence_threshold,
            "last_confidence": self._last_confidence,
            "q_checkpoint": str(self._q_checkpoint_path) if self._q_checkpoint_path else None,
            "q_checkpoint_exists": bool(self._q_checkpoint_path and self._q_checkpoint_path.is_file()),
            "q_top_k": self._q_top_k,
            "q_uncertainty_penalty": self._q_uncertainty_penalty,
            "q_min_margin": self._q_min_margin,
            "q_max_uncertainty": self._q_max_uncertainty,
            "q_max_override_rate": self._q_max_override_rate,
            "q_override_credit": self._q_override_credit,
            "q_actual_override_rate": (
                self._diagnostics["q_overrides"] / self._diagnostics["q_actions"]
                if self._diagnostics["q_actions"] else 0.0
            ),
            "q_min_validation_rows": self._q_min_validation_rows,
            "q_max_validation_mae": (
                self._q_max_validation_mae if math.isfinite(self._q_max_validation_mae) else None
            ),
            "last_q_margin": self._last_q_margin,
            "last_q_uncertainty": self._last_q_uncertainty,
        })
        return result
