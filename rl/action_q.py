from __future__ import annotations

import torch
from torch import nn


class ActionValueEnsemble(nn.Module):
    """Small action-conditioned Q(s,a) ensemble over frozen actor features."""

    def __init__(self, hidden_dim: int, heads: int = 3) -> None:
        super().__init__()
        if hidden_dim <= 0 or heads <= 0:
            raise ValueError("action-Q dimensions must be positive")
        self.hidden_dim = int(hidden_dim)
        self.head_count = int(heads)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(heads)
        ])

    def forward(self, state: torch.Tensor, encoded_options: torch.Tensor) -> torch.Tensor:
        expanded = state[:, None, :].expand_as(encoded_options)
        features = torch.cat((expanded, encoded_options, expanded * encoded_options), dim=-1)
        return torch.stack([head(features).squeeze(-1) for head in self.heads], dim=-1)


def q_mean_and_std(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 3:
        raise ValueError("action-Q values must have [batch, option, head] shape")
    return values.mean(dim=-1), values.std(dim=-1, unbiased=False)
