from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from model import TransformerBlock


@dataclass(frozen=True)
class DeckIdentityModelConfig:
    state_dim: int = 320
    option_dim: int = 176
    entity_num_dim: int = 12
    card_vocab: int = 1600
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ff_dim: int = 384
    max_actions: int = 40
    history_length: int = 8
    deck_size: int = 60
    opponent_classes: int = 7
    dropout: float = 0.05
    layer_norm_eps: float = 1e-5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PTCGDeckIdentityTransformerPolicy(nn.Module):
    """Sequence policy conditioned on exact own deck and visible opponent-card evidence."""

    CAT_VOCABS = (1600, 18, 3, 6, 65, 18, 65, 8, 14, 4)

    def __init__(self, config: DeckIdentityModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        vocabs = list(self.CAT_VOCABS)
        vocabs[0] = config.card_vocab
        self.state_projection = nn.Linear(config.state_dim, d)
        self.history_state_projection = nn.Linear(config.state_dim, d)
        self.history_action_projection = nn.Linear(config.option_dim, d)
        self.history_position = nn.Embedding(config.history_length, d)
        self.option_projection = nn.Linear(config.option_dim, d)
        self.entity_embeddings = nn.ModuleList(nn.Embedding(size, d) for size in vocabs)
        self.entity_num_projection = nn.Linear(config.entity_num_dim, d)
        self.token_type = nn.Embedding(6, d)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.option_head = nn.Linear(d, 1)
        self.count_head = nn.Linear(d, config.max_actions + 1)
        self.value_head = nn.Linear(d, 1)
        self.opponent_deck_head = nn.Linear(d, config.opponent_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        state: torch.Tensor,
        history_state: torch.Tensor,
        history_action: torch.Tensor,
        history_mask: torch.Tensor,
        own_deck_cards: torch.Tensor,
        entity_cat: torch.Tensor,
        entity_num: torch.Tensor,
        entity_mask: torch.Tensor,
        options: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        history_length = history_state.shape[1]
        if history_length != self.config.history_length:
            raise ValueError(
                f"expected {self.config.history_length} history slots, got {history_length}"
            )
        if own_deck_cards.shape[1] != self.config.deck_size:
            raise ValueError(f"expected {self.config.deck_size} own deck cards")

        global_token = self.state_projection(state) + self.token_type.weight[0]
        positions = torch.arange(history_length, device=state.device)
        history_token = (
            self.history_state_projection(history_state)
            + self.history_action_projection(history_action)
            + self.history_position(positions)[None, :, :]
            + self.token_type.weight[1]
        )
        own_deck_token = (
            self.entity_embeddings[0](own_deck_cards).mean(dim=1)
            + self.token_type.weight[2]
        )
        entity_base = self.entity_num_projection(entity_num)
        for field, embedding in enumerate(self.entity_embeddings):
            entity_base = entity_base + embedding(entity_cat[:, :, field])

        opponent_mask = entity_mask.bool() & (entity_cat[:, :, 2] == 1)
        opponent_count = opponent_mask.sum(dim=1, keepdim=True).clamp_min(1)
        opponent_summary = (
            (entity_base * opponent_mask[:, :, None]).sum(dim=1)
            / opponent_count.to(entity_base.dtype)
            + self.token_type.weight[3]
        )
        entity_token = entity_base + self.token_type.weight[4]
        option_token = self.option_projection(options) + self.token_type.weight[5]
        x = torch.cat(
            (
                global_token[:, None, :],
                history_token,
                own_deck_token[:, None, :],
                opponent_summary[:, None, :],
                entity_token,
                option_token,
            ),
            dim=1,
        )
        valid = torch.cat(
            (
                torch.ones((batch, 1), dtype=torch.bool, device=state.device),
                history_mask.bool(),
                torch.ones((batch, 2), dtype=torch.bool, device=state.device),
                entity_mask.bool(),
                option_mask.bool(),
            ),
            dim=1,
        )
        for block in self.blocks:
            x = block(x, valid)
        x = self.final_norm(x)
        own_deck_index = 1 + history_length
        opponent_index = own_deck_index + 1
        action_begin = opponent_index + 1 + entity_cat.shape[1]
        option_hidden = x[:, action_begin : action_begin + options.shape[1]]
        option_logits = self.option_head(option_hidden).squeeze(-1)
        option_logits = option_logits.masked_fill(~option_mask.bool(), -1e4)
        count_logits = self.count_head(x[:, 0])
        value_logits = self.value_head(x[:, 0]).squeeze(-1)
        opponent_logits = self.opponent_deck_head(x[:, opponent_index])
        return option_logits, count_logits, value_logits, opponent_logits

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
