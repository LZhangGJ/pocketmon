from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from model import TransformerBlock
from sequence_model import PTCGSequenceTransformerPolicy


@dataclass(frozen=True)
class PublicEventModelConfig:
    state_dim: int = 320
    option_dim: int = 176
    entity_num_dim: int = 12
    event_num_dim: int = 4
    card_vocab: int = 1600
    attack_vocab: int = 1600
    event_type_vocab: int = 32
    event_area_vocab: int = 16
    event_flag_vocab: int = 3
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ff_dim: int = 384
    max_actions: int = 40
    history_length: int = 8
    opponent_event_length: int = 16
    dropout: float = 0.05
    layer_norm_eps: float = 1e-5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PTCGPublicEventTransformerPolicy(nn.Module):
    """Own decision history plus leak-free opponent public-event history."""

    CAT_VOCABS = PTCGSequenceTransformerPolicy.CAT_VOCABS
    EVENT_CARD_FIELDS = 6
    EVENT_FLAG_FIELDS = 3

    def __init__(self, config: PublicEventModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        vocabs = list(self.CAT_VOCABS)
        vocabs[0] = config.card_vocab
        self.state_projection = nn.Linear(config.state_dim, d)
        self.history_state_projection = nn.Linear(config.state_dim, d)
        self.history_action_projection = nn.Linear(config.option_dim, d)
        self.history_position = nn.Embedding(config.history_length, d)
        self.event_type_embedding = nn.Embedding(config.event_type_vocab, d)
        self.event_card_embedding = nn.Embedding(config.card_vocab, d)
        self.event_card_field_embedding = nn.Embedding(self.EVENT_CARD_FIELDS, d)
        self.event_attack_embedding = nn.Embedding(config.attack_vocab, d)
        self.event_from_area_embedding = nn.Embedding(config.event_area_vocab, d)
        self.event_to_area_embedding = nn.Embedding(config.event_area_vocab, d)
        self.event_flag_embeddings = nn.ModuleList(
            nn.Embedding(config.event_flag_vocab, d)
            for _ in range(self.EVENT_FLAG_FIELDS)
        )
        self.event_num_projection = nn.Linear(config.event_num_dim, d)
        self.event_position = nn.Embedding(config.opponent_event_length, d)
        self.option_projection = nn.Linear(config.option_dim, d)
        self.entity_embeddings = nn.ModuleList(nn.Embedding(size, d) for size in vocabs)
        self.entity_num_projection = nn.Linear(config.entity_num_dim, d)
        self.token_type = nn.Embedding(5, d)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.option_head = nn.Linear(d, 1)
        self.count_head = nn.Linear(d, config.max_actions + 1)
        self.value_head = nn.Linear(d, 1)
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
        opponent_event_cat: torch.Tensor,
        opponent_event_num: torch.Tensor,
        opponent_event_mask: torch.Tensor,
        entity_cat: torch.Tensor,
        entity_num: torch.Tensor,
        entity_mask: torch.Tensor,
        options: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        history_length = history_state.shape[1]
        event_length = opponent_event_cat.shape[1]
        if history_length != self.config.history_length:
            raise ValueError(f"expected {self.config.history_length} history slots")
        if event_length != self.config.opponent_event_length:
            raise ValueError(f"expected {self.config.opponent_event_length} event slots")
        global_token = self.state_projection(state) + self.token_type.weight[0]
        history_positions = torch.arange(history_length, device=state.device)
        history_token = (
            self.history_state_projection(history_state)
            + self.history_action_projection(history_action)
            + self.history_position(history_positions)[None, :, :]
            + self.token_type.weight[1]
        )
        event_token = self.event_type_embedding(opponent_event_cat[:, :, 0])
        card_values = opponent_event_cat[:, :, 1:7]
        card_embeddings = self.event_card_embedding(card_values)
        card_fields = self.event_card_field_embedding(
            torch.arange(self.EVENT_CARD_FIELDS, device=state.device)
        )[None, None, :, :]
        card_present = (card_values > 0).unsqueeze(-1)
        event_token = event_token + ((card_embeddings + card_fields) * card_present).sum(dim=2)
        event_token = event_token + self.event_attack_embedding(opponent_event_cat[:, :, 7])
        event_token = event_token + self.event_from_area_embedding(opponent_event_cat[:, :, 8])
        event_token = event_token + self.event_to_area_embedding(opponent_event_cat[:, :, 9])
        for field, embedding in enumerate(self.event_flag_embeddings):
            event_token = event_token + embedding(opponent_event_cat[:, :, 10 + field])
        event_positions = torch.arange(event_length, device=state.device)
        event_token = (
            event_token
            + self.event_num_projection(opponent_event_num)
            + self.event_position(event_positions)[None, :, :]
            + self.token_type.weight[2]
        )
        entity_token = self.entity_num_projection(entity_num)
        for field, embedding in enumerate(self.entity_embeddings):
            entity_token = entity_token + embedding(entity_cat[:, :, field])
        entity_token = entity_token + self.token_type.weight[3]
        option_token = self.option_projection(options) + self.token_type.weight[4]
        x = torch.cat(
            (
                global_token[:, None, :],
                history_token,
                event_token,
                entity_token,
                option_token,
            ),
            dim=1,
        )
        valid = torch.cat(
            (
                torch.ones((batch, 1), dtype=torch.bool, device=state.device),
                history_mask.bool(),
                opponent_event_mask.bool(),
                entity_mask.bool(),
                option_mask.bool(),
            ),
            dim=1,
        )
        for block in self.blocks:
            x = block(x, valid)
        x = self.final_norm(x)
        action_begin = 1 + history_length + event_length + entity_cat.shape[1]
        option_hidden = x[:, action_begin : action_begin + options.shape[1]]
        option_logits = self.option_head(option_hidden).squeeze(-1)
        option_logits = option_logits.masked_fill(~option_mask.bool(), -1e4)
        count_logits = self.count_head(x[:, 0])
        value_logits = self.value_head(x[:, 0]).squeeze(-1)
        return option_logits, count_logits, value_logits

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
