from __future__ import annotations

import torch
from torch import nn

from .features import (
    ACTION_DIM,
    ATTACK_METADATA_DIM,
    ATTACK_VOCAB_SIZE,
    CARD_METADATA_DIM,
    CARD_TEXT_EMBEDDING_DIM,
    CARD_VOCAB_SIZE,
    BELIEF_DIM,
    ENTITY_DIM,
    ENTITY_ZONE_COUNT,
    HISTORY_DIM,
    RESOURCE_DIM,
    STATE_DIM,
)


def legal_choice_mask(
    option_mask: torch.Tensor,
    selected_mask: torch.Tensor,
    selected_count: torch.Tensor,
    min_count: torch.Tensor,
    max_count: torch.Tensor,
) -> torch.Tensor:
    """Return the legal candidate-plus-STOP mask for an autoregressive step."""

    candidates = option_mask & ~selected_mask & (selected_count[:, None] < max_count[:, None])
    stop = (selected_count >= min_count) & (selected_count <= max_count)
    return torch.cat((candidates, stop[:, None]), dim=1)


class MaskedPointerActorCritic(nn.Module):
    """Candidate pointer with optional causal GRU history and an explicit STOP action."""

    def __init__(self, hidden_dim: int = 128, history_encoder: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.history_encoder_enabled = bool(history_encoder)
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, hidden_dim), nn.Tanh())
        if self.history_encoder_enabled:
            self.history_encoder = nn.GRU(HISTORY_DIM, hidden_dim, batch_first=True)
            self.history_projection = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.option_encoder = nn.Sequential(nn.Linear(ACTION_DIM, hidden_dim), nn.Tanh())
        self.selection_encoder = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.step_encoder = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh())
        self.policy = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.stop_policy = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def encode_history(self, histories: torch.Tensor, history_lengths: torch.Tensor) -> torch.Tensor:
        if not self.history_encoder_enabled:
            return torch.zeros(
                (histories.shape[0], self.hidden_dim), dtype=histories.dtype, device=histories.device
            )
        output, _ = self.history_encoder(histories)
        safe_indices = history_lengths.clamp_min(1) - 1
        batch_indices = torch.arange(histories.shape[0], device=histories.device)
        encoded = output[batch_indices, safe_indices]
        encoded = torch.where(history_lengths[:, None] > 0, encoded, torch.zeros_like(encoded))
        return self.history_projection(encoded)

    def encode(
        self,
        states: torch.Tensor,
        options: torch.Tensor,
        histories: torch.Tensor | None = None,
        history_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.state_encoder(states)
        if self.history_encoder_enabled:
            if histories is None or history_lengths is None:
                raise ValueError("history-enabled model requires histories and history_lengths")
            state = state + self.encode_history(histories, history_lengths)
        encoded_options = self.option_encoder(options)
        return state, encoded_options, self.value(state).squeeze(-1)

    def encode_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encode(
            batch["states"], batch["options"], batch["histories"], batch["history_lengths"]
        )

    def pointer_logits(
        self,
        state: torch.Tensor,
        encoded_options: torch.Tensor,
        selected_mask: torch.Tensor,
        selected_count: torch.Tensor,
    ) -> torch.Tensor:
        weights = selected_mask.to(encoded_options.dtype)
        selected_sum = (encoded_options * weights[:, :, None]).sum(dim=1)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        selected = self.selection_encoder(selected_sum / denominator)
        step = self.step_encoder(selected_count.to(state.dtype)[:, None] / 20.0)
        context = state + step
        expanded_context = context[:, None, :].expand(-1, encoded_options.shape[1], -1)
        expanded_selected = selected[:, None, :].expand_as(expanded_context)
        option_logits = self.policy(torch.cat((expanded_context, expanded_selected, encoded_options), dim=-1)).squeeze(-1)
        stop_logit = self.stop_policy(torch.cat((context, selected), dim=-1))
        return torch.cat((option_logits, stop_logit), dim=1)

    def forward(
        self,
        states: torch.Tensor,
        options: torch.Tensor,
        histories: torch.Tensor | None = None,
        history_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible single-step option scores (without STOP)."""

        state, encoded_options, values = self.encode(states, options, histories, history_lengths)
        selected = torch.zeros(options.shape[:2], dtype=torch.bool, device=options.device)
        counts = torch.zeros(options.shape[0], dtype=torch.long, device=options.device)
        return self.pointer_logits(state, encoded_options, selected, counts)[:, :-1], values


# Keep the old public name for callers while making the new decoder explicit.
CandidateActorCritic = MaskedPointerActorCritic


class StructuredMaskedPointerActorCritic(MaskedPointerActorCritic):
    """Pointer policy conditioned on card/attack IDs, visible entities and own deck."""

    def __init__(
        self,
        hidden_dim: int = 192,
        card_metadata: torch.Tensor | None = None,
        attack_metadata: torch.Tensor | None = None,
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, history_encoder=False)
        card_metadata = (
            torch.zeros(CARD_VOCAB_SIZE, CARD_METADATA_DIM)
            if card_metadata is None else card_metadata.to(dtype=torch.float32)
        )
        attack_metadata = (
            torch.zeros(ATTACK_VOCAB_SIZE, ATTACK_METADATA_DIM)
            if attack_metadata is None else attack_metadata.to(dtype=torch.float32)
        )
        if tuple(card_metadata.shape) != (CARD_VOCAB_SIZE, CARD_METADATA_DIM):
            raise ValueError("unexpected card metadata table shape")
        if tuple(attack_metadata.shape) != (ATTACK_VOCAB_SIZE, ATTACK_METADATA_DIM):
            raise ValueError("unexpected attack metadata table shape")
        self.register_buffer("card_metadata", card_metadata)
        self.register_buffer("attack_metadata", attack_metadata)
        self.card_embedding = nn.Embedding(CARD_VOCAB_SIZE, 32, padding_idx=0)
        self.attack_embedding = nn.Embedding(ATTACK_VOCAB_SIZE, 24, padding_idx=0)
        self.zone_embedding = nn.Embedding(ENTITY_ZONE_COUNT, 12, padding_idx=0)
        self.card_metadata_encoder = nn.Sequential(nn.Linear(CARD_METADATA_DIM, 16), nn.Tanh())
        self.attack_metadata_encoder = nn.Sequential(nn.Linear(ATTACK_METADATA_DIM, 16), nn.Tanh())
        self.option_identity_encoder = nn.Sequential(nn.Linear((32 + 16) * 2 + 24 + 16, hidden_dim), nn.Tanh())
        self.entity_encoder = nn.Sequential(nn.Linear(32 + 16 + 12 + ENTITY_DIM, hidden_dim), nn.Tanh())
        self.entity_pool = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh())
        self.deck_encoder = nn.Sequential(nn.Linear(32 + 16, hidden_dim), nn.Tanh())
        self.context_norm = nn.LayerNorm(hidden_dim)

    def _card_repr(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.cat((
            self.card_embedding(ids),
            self.card_metadata_encoder(self.card_metadata[ids]),
        ), dim=-1)

    def _attack_repr(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.cat((
            self.attack_embedding(ids),
            self.attack_metadata_encoder(self.attack_metadata[ids]),
        ), dim=-1)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(values.dtype)[..., None]
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = values.masked_fill(~mask[..., None], -torch.inf)
        pooled = masked.max(dim=1).values
        return torch.where(mask.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))

    def encode_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.state_encoder(batch["states"])

        option_identity = torch.cat((
            self._card_repr(batch["option_card_ids"]),
            self._card_repr(batch["option_target_card_ids"]),
            self._attack_repr(batch["option_attack_ids"]),
        ), dim=-1)
        encoded_options = self.option_encoder(batch["options"]) + self.option_identity_encoder(option_identity)

        entities = self.entity_encoder(torch.cat((
            self._card_repr(batch["entity_card_ids"]),
            self.zone_embedding(batch["entity_zone_ids"]),
            batch["entity_features"],
        ), dim=-1))
        entity_context = self.entity_pool(torch.cat((
            self._masked_mean(entities, batch["entity_mask"]),
            self._masked_max(entities, batch["entity_mask"]),
        ), dim=-1))

        deck_cards = self._card_repr(batch["deck_card_ids"])
        deck_context = self.deck_encoder(self._masked_mean(deck_cards, batch["deck_mask"]))
        state = self.context_norm(state + entity_context + deck_context)
        return state, encoded_options, self.value(state).squeeze(-1)


class StructuredTransformerMaskedPointerActorCritic(StructuredMaskedPointerActorCritic):
    """Card-identity Transformer with official effect-text conditioning."""

    def __init__(
        self,
        hidden_dim: int = 192,
        card_metadata: torch.Tensor | None = None,
        attack_metadata: torch.Tensor | None = None,
        card_text_embeddings: torch.Tensor | None = None,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ) -> None:
        if hidden_dim % transformer_heads:
            raise ValueError("hidden_dim must be divisible by transformer_heads")
        super().__init__(
            hidden_dim=hidden_dim,
            card_metadata=card_metadata,
            attack_metadata=attack_metadata,
        )
        card_text_embeddings = (
            torch.zeros(CARD_VOCAB_SIZE, CARD_TEXT_EMBEDDING_DIM)
            if card_text_embeddings is None else card_text_embeddings.to(dtype=torch.float32)
        )
        if tuple(card_text_embeddings.shape) != (CARD_VOCAB_SIZE, CARD_TEXT_EMBEDDING_DIM):
            raise ValueError("unexpected card text embedding table shape")
        self.register_buffer("card_text_embeddings", card_text_embeddings)
        self.card_text_encoder = nn.Sequential(nn.Linear(CARD_TEXT_EMBEDDING_DIM, 16), nn.Tanh())

        card_repr_dim = 32 + 16 + 16
        attack_repr_dim = 24 + 16
        self.option_identity_encoder = nn.Sequential(
            nn.Linear(card_repr_dim * 2 + attack_repr_dim, hidden_dim), nn.Tanh()
        )
        self.entity_encoder = nn.Sequential(
            nn.Linear(card_repr_dim + 12 + ENTITY_DIM, hidden_dim), nn.Tanh()
        )
        self.deck_encoder = nn.Sequential(nn.Linear(card_repr_dim, hidden_dim), nn.Tanh())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.entity_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.option_entity_attention = nn.MultiheadAttention(
            hidden_dim, transformer_heads, dropout=0.0, batch_first=True
        )
        self.entity_context_norm = nn.LayerNorm(hidden_dim)
        self.option_context_norm = nn.LayerNorm(hidden_dim)

    def _card_repr(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.cat((
            self.card_embedding(ids),
            self.card_metadata_encoder(self.card_metadata[ids]),
            self.card_text_encoder(self.card_text_embeddings[ids]),
        ), dim=-1)

    def encode_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.state_encoder(batch["states"])
        option_identity = torch.cat((
            self._card_repr(batch["option_card_ids"]),
            self._card_repr(batch["option_target_card_ids"]),
            self._attack_repr(batch["option_attack_ids"]),
        ), dim=-1)
        encoded_options = self.option_encoder(batch["options"]) + self.option_identity_encoder(option_identity)

        entities = self.entity_encoder(torch.cat((
            self._card_repr(batch["entity_card_ids"]),
            self.zone_embedding(batch["entity_zone_ids"]),
            batch["entity_features"],
        ), dim=-1))
        entity_mask = batch["entity_mask"].clone()
        empty_rows = ~entity_mask.any(dim=1)
        entity_mask[empty_rows, 0] = True
        entities = entities.masked_fill(~entity_mask[..., None], 0.0)
        entities = self.entity_transformer(entities, src_key_padding_mask=~entity_mask)
        entity_context = self.entity_context_norm(self._masked_mean(entities, entity_mask))

        deck_cards = self._card_repr(batch["deck_card_ids"])
        deck_context = self.deck_encoder(self._masked_mean(deck_cards, batch["deck_mask"]))
        state = self.context_norm(state + entity_context + deck_context)

        option_queries = encoded_options + state[:, None, :] + deck_context[:, None, :]
        attended, _ = self.option_entity_attention(
            option_queries,
            entities,
            entities,
            key_padding_mask=~entity_mask,
            need_weights=False,
        )
        encoded_options = self.option_context_norm(option_queries + attended)
        return state, encoded_options, self.value(state).squeeze(-1)


class TemporalResourceBeliefTransformerActorCritic(StructuredTransformerMaskedPointerActorCritic):
    """Structured Transformer with independently switchable causal contexts.

    Keeping the three contexts switchable makes the Gold V3.1 experiments true
    one-factor ablations while preserving one checkpoint architecture.
    """

    def __init__(
        self,
        hidden_dim: int = 192,
        card_metadata: torch.Tensor | None = None,
        attack_metadata: torch.Tensor | None = None,
        card_text_embeddings: torch.Tensor | None = None,
        *,
        history_length: int = 32,
        use_history: bool = True,
        use_resources: bool = True,
        use_opponent_belief: bool = True,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            card_metadata=card_metadata,
            attack_metadata=attack_metadata,
            card_text_embeddings=card_text_embeddings,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
        )
        if history_length <= 0 or history_length > 128:
            raise ValueError("history_length must be in [1, 128]")
        self.history_length = int(history_length)
        self.use_history = bool(use_history)
        self.use_resources = bool(use_resources)
        self.use_opponent_belief = bool(use_opponent_belief)
        self.history_token_encoder = nn.Sequential(nn.Linear(HISTORY_DIM, hidden_dim), nn.Tanh())
        self.history_position = nn.Embedding(self.history_length, hidden_dim)
        history_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.history_transformer = nn.TransformerEncoder(history_layer, num_layers=2)
        card_repr_dim = 32 + 16 + 16
        self.remaining_encoder = nn.Sequential(nn.Linear(card_repr_dim, hidden_dim), nn.Tanh())
        self.resource_encoder = nn.Sequential(nn.Linear(RESOURCE_DIM, hidden_dim), nn.Tanh())
        self.belief_deck_encoder = nn.Sequential(nn.Linear(card_repr_dim, hidden_dim), nn.Tanh())
        self.belief_feature_encoder = nn.Sequential(nn.Linear(BELIEF_DIM, hidden_dim), nn.Tanh())
        self.v31_context_norm = nn.LayerNorm(hidden_dim)
        self.v31_option_norm = nn.LayerNorm(hidden_dim)

    def _history_context(self, batch: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        if not self.use_history:
            return torch.zeros_like(state)
        histories = batch["histories"][:, -self.history_length:]
        mask = batch["history_mask"][:, -self.history_length:].clone()
        empty = ~mask.any(dim=1)
        mask[empty, 0] = True
        positions = torch.arange(histories.shape[1], device=histories.device)
        tokens = self.history_token_encoder(histories) + self.history_position(positions)[None, :, :]
        tokens = tokens.masked_fill(~mask[..., None], 0.0)
        causal = torch.triu(
            torch.ones((histories.shape[1], histories.shape[1]), dtype=torch.bool, device=histories.device),
            diagonal=1,
        )
        encoded = self.history_transformer(tokens, mask=causal, src_key_padding_mask=~mask)
        lengths = mask.sum(dim=1).clamp_min(1) - 1
        context = encoded[torch.arange(len(encoded), device=encoded.device), lengths]
        return torch.where(empty[:, None], torch.zeros_like(context), context)

    def _card_context(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        encoder: nn.Module,
    ) -> torch.Tensor:
        safe_mask = mask.clone()
        safe_mask[~safe_mask.any(dim=1), 0] = True
        encoded = encoder(self._card_repr(ids))
        encoded = encoded.masked_fill(~safe_mask[..., None], 0.0)
        return self._masked_mean(encoded, safe_mask)

    def encode_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, encoded_options, _ = super().encode_batch(batch)
        context = self._history_context(batch, state)
        if self.use_resources:
            context = context + self._card_context(
                batch["remaining_card_ids"], batch["remaining_card_mask"], self.remaining_encoder
            ) + self.resource_encoder(batch["resource_features"])
        if self.use_opponent_belief:
            context = context + self._card_context(
                batch["opponent_belief_card_ids"],
                batch["opponent_belief_card_mask"],
                self.belief_deck_encoder,
            ) + self.belief_feature_encoder(batch["opponent_belief_features"])
        state = self.v31_context_norm(state + context)
        encoded_options = self.v31_option_norm(encoded_options + context[:, None, :])
        return state, encoded_options, self.value(state).squeeze(-1)
