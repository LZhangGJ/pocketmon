from __future__ import annotations

import torch
from torch import nn

from .features import ACTION_DIM, STATE_DIM


class CandidateActorCritic(nn.Module):
    """Scores variable legal options and estimates the current state value."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM, hidden_dim), nn.Tanh())
        self.option_encoder = nn.Sequential(nn.Linear(ACTION_DIM, hidden_dim), nn.Tanh())
        self.policy = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, states: torch.Tensor, options: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # states: [B, S], options: [B, N, A]
        state = self.state_encoder(states)
        encoded_options = self.option_encoder(options)
        expanded_state = state[:, None, :].expand(-1, options.shape[1], -1)
        logits = self.policy(torch.cat((expanded_state, encoded_options), dim=-1)).squeeze(-1)
        return logits, self.value(state).squeeze(-1)
