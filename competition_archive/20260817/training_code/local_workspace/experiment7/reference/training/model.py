from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


try:
    F.gelu(torch.zeros(1), approximate="tanh")
except TypeError:
    def _gelu_tanh(value: torch.Tensor) -> torch.Tensor:
        # torch<1.12 lacks the ``approximate`` keyword.  This is the formula
        # used by the native tanh approximation, kept here for portable
        # validation workers with an older but otherwise compatible runtime.
        return 0.5 * value * (
            1.0
            + torch.tanh(
                0.7978845608028654 * (value + 0.044715 * torch.pow(value, 3))
            )
        )
else:
    def _gelu_tanh(value: torch.Tensor) -> torch.Tensor:
        return F.gelu(value, approximate="tanh")


@dataclass(frozen=True)
class ModelConfig:
    state_dim: int = 320
    option_dim: int = 176
    entity_num_dim: int = 12
    card_vocab: int = 1600
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ff_dim: int = 384
    max_actions: int = 40
    dropout: float = 0.05
    layer_norm_eps: float = 1e-5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = d // config.n_heads
        if d % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.ln1 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(d, config.ff_dim)
        self.fc2 = nn.Linear(config.ff_dim, d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        norm = self.ln1(x)
        qkv = self.qkv(norm).reshape(batch, tokens, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores.masked_fill(~valid_mask[:, None, None, :], -1e4)
        attention = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(attention, v).transpose(1, 2).reshape(batch, tokens, width)
        x = x + self.dropout(self.out(mixed))
        hidden = self.fc2(_gelu_tanh(self.fc1(self.ln2(x))))
        x = x + self.dropout(hidden)
        return x


class PTCGTransformerPolicy(nn.Module):
    """Set Transformer over a global state, visible card entities and legal actions."""

    CAT_VOCABS = (1600, 18, 3, 6, 65, 18, 65, 8, 14, 4)

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        vocabs = list(self.CAT_VOCABS)
        vocabs[0] = config.card_vocab
        self.state_projection = nn.Linear(config.state_dim, d)
        self.option_projection = nn.Linear(config.option_dim, d)
        self.entity_embeddings = nn.ModuleList(nn.Embedding(size, d) for size in vocabs)
        self.entity_num_projection = nn.Linear(config.entity_num_dim, d)
        self.token_type = nn.Embedding(3, d)
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
        entity_cat: torch.Tensor,
        entity_num: torch.Tensor,
        entity_mask: torch.Tensor,
        options: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        global_token = self.state_projection(state) + self.token_type.weight[0]
        entity_token = self.entity_num_projection(entity_num)
        for field, embedding in enumerate(self.entity_embeddings):
            entity_token = entity_token + embedding(entity_cat[:, :, field])
        entity_token = entity_token + self.token_type.weight[1]
        option_token = self.option_projection(options) + self.token_type.weight[2]
        x = torch.cat((global_token[:, None, :], entity_token, option_token), dim=1)
        valid = torch.cat(
            (
                torch.ones((batch, 1), dtype=torch.bool, device=state.device),
                entity_mask.bool(),
                option_mask.bool(),
            ),
            dim=1,
        )
        for block in self.blocks:
            x = block(x, valid)
        x = self.final_norm(x)
        action_begin = 1 + entity_cat.shape[1]
        option_hidden = x[:, action_begin : action_begin + options.shape[1]]
        option_logits = self.option_head(option_hidden).squeeze(-1)
        option_logits = option_logits.masked_fill(~option_mask.bool(), -1e4)
        count_logits = self.count_head(x[:, 0])
        value_logits = self.value_head(x[:, 0]).squeeze(-1)
        return option_logits, count_logits, value_logits

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
