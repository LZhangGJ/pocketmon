from __future__ import annotations

import torch
from torch import nn

from .features import ACTION_DIM, HISTORY_DIM, STATE_DIM


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
