from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from .agent_adapter import RLBCPolicyAdapter
from .temporal_model import (
    TEMPORAL_ARCHITECTURE,
    StructuredTemporalTransformerActorCritic,
)


class TemporalRLBCPolicyAdapter(RLBCPolicyAdapter):
    """Runtime adapter for RL-BC-004 temporal checkpoints.

    The base adapter still owns legal decoding, confidence fallback, history
    updates, diagnostics and emergency handling. This subclass only replaces
    checkpoint construction so existing stable runtime behavior remains intact.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        fallback: Callable[[dict[str, Any]], list[int]],
        device: str = "cpu",
        confidence_threshold: float | None = None,
        deck: list[int] | None = None,
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            fallback=fallback,
            device=device,
            confidence_threshold=confidence_threshold,
            deck=deck,
            q_checkpoint_path=None,
        )

    def _load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            try:
                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(
                    self.checkpoint_path,
                    map_location=self.device,
                )
            config = checkpoint["config"]
            architecture = config.get("architecture")
            if architecture != TEMPORAL_ARCHITECTURE:
                raise ValueError(
                    f"unsupported temporal checkpoint architecture: {architecture}"
                )

            self._history_enabled = True
            self._structured = True
            self._history_length = int(config.get("history_length", 0))
            if self._history_length <= 0:
                raise ValueError("temporal checkpoint has no positive history_length")
            configured_threshold = float(config.get("confidence_threshold", 0.0))
            self._confidence_threshold = (
                configured_threshold
                if self._confidence_threshold_override is None
                else float(self._confidence_threshold_override)
            )
            if not 0.0 <= self._confidence_threshold <= 1.0:
                raise ValueError("confidence threshold must be in [0, 1]")

            model = StructuredTemporalTransformerActorCritic(
                hidden_dim=int(checkpoint["hidden_dim"]),
                history_length=self._history_length,
                transformer_layers=int(config.get("transformer_layers", 2)),
                transformer_heads=int(config.get("transformer_heads", 4)),
                transformer_ffn_dim=int(config.get("transformer_ffn_dim", 768)),
                transformer_dropout=float(config.get("transformer_dropout", 0.10)),
            ).to(self.device)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            self._model = model
        except Exception:
            self._diagnostics["load_errors"] += 1
            self._model = None
