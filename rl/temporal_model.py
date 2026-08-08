from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .features import HISTORY_DIM
from .model import StructuredMaskedPointerActorCritic


TEMPORAL_ARCHITECTURE = (
    "structured_card_attack_deepsets_deck_transformer8_masked_pointer_with_stop"
)
TEMPORAL_STATE_PREFIXES = (
    "history_input.",
    "history_positions.",
    "history_encoder.",
    "history_output.",
    "history_gate.",
    "temporal_context_norm.",
)


class StructuredTemporalTransformerActorCritic(StructuredMaskedPointerActorCritic):
    """Structured pointer policy with a bounded causal Transformer history.

    The current observation still uses the existing card/attack/entity/deck
    encoder. Only the most recent ``history_length`` completed decisions are
    passed to the Transformer, keeping CPU inference bounded for Kaggle's first
    round.
    """

    def __init__(
        self,
        hidden_dim: int = 192,
        card_metadata: torch.Tensor | None = None,
        attack_metadata: torch.Tensor | None = None,
        *,
        history_length: int = 8,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ffn_dim: int = 768,
        transformer_dropout: float = 0.10,
    ) -> None:
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        if transformer_layers <= 0:
            raise ValueError("transformer_layers must be positive")
        if transformer_heads <= 0 or hidden_dim % transformer_heads:
            raise ValueError("transformer_heads must divide hidden_dim")
        if transformer_ffn_dim < hidden_dim:
            raise ValueError("transformer_ffn_dim must be at least hidden_dim")
        if not 0.0 <= transformer_dropout < 1.0:
            raise ValueError("transformer_dropout must be in [0, 1)")

        super().__init__(
            hidden_dim=hidden_dim,
            card_metadata=card_metadata,
            attack_metadata=attack_metadata,
        )
        self.history_encoder_enabled = True
        self.history_length = int(history_length)
        self.transformer_layers = int(transformer_layers)
        self.transformer_heads = int(transformer_heads)
        self.transformer_ffn_dim = int(transformer_ffn_dim)
        self.transformer_dropout = float(transformer_dropout)

        self.history_input = nn.Sequential(
            nn.Linear(HISTORY_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.history_positions = nn.Embedding(self.history_length, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=self.transformer_heads,
            dim_feedforward=self.transformer_ffn_dim,
            dropout=self.transformer_dropout,
            activation="gelu",
            batch_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.transformer_layers,
        )
        self.history_output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.history_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.temporal_context_norm = nn.LayerNorm(hidden_dim)

    def encode_history(
        self,
        histories: torch.Tensor,
        history_mask: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if histories.ndim != 3 or histories.shape[-1] != HISTORY_DIM:
            raise ValueError("histories must have shape [batch, time, HISTORY_DIM]")
        if history_mask.shape != histories.shape[:2]:
            raise ValueError("history_mask shape mismatch")
        if history_lengths.shape != histories.shape[:1]:
            raise ValueError("history_lengths shape mismatch")
        if histories.shape[1] > self.history_length:
            raise ValueError(
                f"history tensor length {histories.shape[1]} exceeds configured "
                f"maximum {self.history_length}"
            )

        batch_size, sequence_length, _ = histories.shape
        positions = torch.arange(sequence_length, device=histories.device)
        encoded = self.history_input(histories)
        encoded = encoded + self.history_positions(positions)[None, :, :]

        # Transformer attention over an entirely padded row may produce NaNs on
        # some PyTorch versions. Temporarily expose one zero-history token, then
        # explicitly zero the pooled result for truly empty rows.
        safe_mask = history_mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if bool(empty_rows.any()):
            safe_mask[empty_rows, 0] = True
        encoded = encoded.masked_fill(~safe_mask[:, :, None], 0.0)
        encoded = self.history_encoder(
            encoded,
            src_key_padding_mask=~safe_mask,
        )

        safe_indices = history_lengths.clamp(min=1, max=sequence_length) - 1
        batch_indices = torch.arange(batch_size, device=histories.device)
        pooled = encoded[batch_indices, safe_indices]
        pooled = torch.where(
            history_lengths[:, None] > 0,
            pooled,
            torch.zeros_like(pooled),
        )
        return self.history_output(pooled)

    def encode_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_state, encoded_options, _ = super().encode_batch(batch)
        temporal_state = self.encode_history(
            batch["histories"],
            batch["history_mask"],
            batch["history_lengths"],
        )
        gate = torch.sigmoid(
            self.history_gate(torch.cat((current_state, temporal_state), dim=-1))
        )
        fused_state = self.temporal_context_norm(
            current_state + gate * temporal_state
        )
        return fused_state, encoded_options, self.value(fused_state).squeeze(-1)


def load_structured_warm_start(
    model: StructuredTemporalTransformerActorCritic,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, list[str]]:
    """Load a structured BC checkpoint while requiring all base weights.

    Only the newly introduced temporal modules may be absent. This prevents a
    typo or incompatible checkpoint from silently leaving existing policy
    modules randomly initialized.
    """

    incompatible = model.load_state_dict(dict(state_dict), strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    invalid_missing = [
        key for key in missing
        if not key.startswith(TEMPORAL_STATE_PREFIXES)
    ]
    if unexpected or invalid_missing:
        raise ValueError(
            "incompatible structured warm start: "
            f"unexpected={unexpected}, invalid_missing={invalid_missing}"
        )
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
