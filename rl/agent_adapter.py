from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import torch

from .bc import action_is_legal, collate_rows, greedy_decode_with_confidence
from .features import action_features, history_features, state_features, structured_observation_features
from .model import MaskedPointerActorCritic, StructuredMaskedPointerActorCritic


STATELESS_ARCHITECTURE = "stateless_masked_autoregressive_candidate_pointer_with_stop"
HISTORY_ARCHITECTURE = "causal_gru_history_masked_autoregressive_candidate_pointer_with_stop"
STRUCTURED_ARCHITECTURE = "structured_card_attack_deepsets_deck_masked_pointer_with_stop"


class RLBCPolicyAdapter:
    """Local-game adapter with masked decode and a validated rule fallback."""

    def __init__(
        self,
        checkpoint_path: Path,
        fallback: Callable[[dict[str, Any]], list[int]],
        device: str = "cpu",
        confidence_threshold: float | None = None,
        deck: list[int] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.fallback = fallback
        self.device = torch.device(device)
        self._model: MaskedPointerActorCritic | None = None
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
            if architecture not in (STATELESS_ARCHITECTURE, HISTORY_ARCHITECTURE, STRUCTURED_ARCHITECTURE):
                raise ValueError(f"unsupported checkpoint architecture: {architecture}")
            self._history_enabled = architecture == HISTORY_ARCHITECTURE
            self._structured = architecture == STRUCTURED_ARCHITECTURE
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
            model = (
                StructuredMaskedPointerActorCritic(int(checkpoint["hidden_dim"]))
                if self._structured else
                MaskedPointerActorCritic(int(checkpoint["hidden_dim"]), history_encoder=self._history_enabled)
            ).to(self.device)
            model.load_state_dict(checkpoint["model"])
            model.eval()
            self._model = model
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
        })
        return result
