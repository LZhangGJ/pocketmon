from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from model import TransformerBlock


GREEDY_TIE_TOLERANCE = 5e-4


def stable_torch_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Choose the lowest index inside the portable runtime's near-tie band."""

    maximum = logits.max(dim=1, keepdim=True).values
    near_maximum = logits >= maximum - GREEDY_TIE_TOLERANCE
    indices = torch.arange(logits.shape[1], device=logits.device).expand_as(logits)
    sentinel = torch.full_like(indices, logits.shape[1])
    return torch.where(near_maximum, indices, sentinel).min(dim=1).values


@dataclass(frozen=True)
class UniversalDeckModelConfig:
    """Experiment 7 Transformer with a set-valued deck encoder and joint policy."""

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
    deck_latents: int = 8
    opponent_classes: int = 1
    dropout: float = 0.05
    layer_norm_eps: float = 1e-5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UniversalDeckEncoding:
    global_hidden: torch.Tensor
    option_hidden: torch.Tensor
    option_mask: torch.Tensor
    deck_hidden: torch.Tensor
    value_logits: torch.Tensor
    opponent_logits: torch.Tensor


class UniversalDeckTransformerPolicy(nn.Module):
    """Universal BC/PPO policy with eight learned deck tokens.

    The 60-card deck is treated as a multiset. Learned latent queries attend to
    card embeddings without positional encodings, so order is irrelevant while
    duplicate-card counts remain observable. Selection is an autoregressive
    distribution over remaining options plus STOP; it cannot emit an illegal
    count or select an option twice.
    """

    CAT_VOCABS = (1600, 18, 3, 6, 65, 18, 65, 8, 14, 4)

    def __init__(self, config: UniversalDeckModelConfig) -> None:
        super().__init__()
        if config.deck_latents < 1:
            raise ValueError("deck_latents must be positive")
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
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

        self.deck_queries = nn.Parameter(torch.empty(config.deck_latents, d))
        self.deck_attention = nn.MultiheadAttention(
            d, config.n_heads, dropout=config.dropout, batch_first=True
        )
        self.deck_norm1 = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.deck_ff = nn.Sequential(nn.Linear(d, config.ff_dim), nn.GELU(), nn.Linear(config.ff_dim, d))
        self.deck_norm2 = nn.LayerNorm(d, eps=config.layer_norm_eps)

        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(d, eps=config.layer_norm_eps)
        self.value_head = nn.Linear(d, 1)
        self.opponent_deck_head = nn.Linear(d, config.opponent_classes)

        self.decoder_step = nn.Embedding(config.max_actions + 1, d)
        self.decoder_context = nn.Linear(3 * d, d)
        self.decoder_option = nn.Linear(d, d, bias=False)
        self.decoder_option_bias = nn.Linear(d, 1)
        self.decoder_stop = nn.Linear(d, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        nn.init.normal_(self.deck_queries, std=0.02)

    def _deck_tokens(self, own_deck_cards: torch.Tensor) -> torch.Tensor:
        if own_deck_cards.ndim != 2 or own_deck_cards.shape[1] != self.config.deck_size:
            raise ValueError(f"expected [batch,{self.config.deck_size}] own deck cards")
        cards = self.entity_embeddings[0](own_deck_cards)
        queries = self.deck_queries[None, :, :].expand(cards.shape[0], -1, -1)
        attended, _ = self.deck_attention(queries, cards, cards, need_weights=False)
        hidden = self.deck_norm1(queries + attended)
        return self.deck_norm2(hidden + self.deck_ff(hidden))

    def encode(
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
    ) -> UniversalDeckEncoding:
        batch = state.shape[0]
        history_length = history_state.shape[1]
        if history_length != self.config.history_length:
            raise ValueError(f"expected {self.config.history_length} history slots, got {history_length}")

        global_token = self.state_projection(state) + self.token_type.weight[0]
        positions = torch.arange(history_length, device=state.device)
        history_token = (
            self.history_state_projection(history_state)
            + self.history_action_projection(history_action)
            + self.history_position(positions)[None, :, :]
            + self.token_type.weight[1]
        )
        deck_token = self._deck_tokens(own_deck_cards) + self.token_type.weight[2]

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
                deck_token,
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
                torch.ones(
                    (batch, self.config.deck_latents + 1),
                    dtype=torch.bool,
                    device=state.device,
                ),
                entity_mask.bool(),
                option_mask.bool(),
            ),
            dim=1,
        )
        for block in self.blocks:
            x = block(x, valid)
        x = self.final_norm(x)

        deck_begin = 1 + history_length
        opponent_index = deck_begin + self.config.deck_latents
        action_begin = opponent_index + 1 + entity_cat.shape[1]
        global_hidden = x[:, 0]
        option_hidden = x[:, action_begin : action_begin + options.shape[1]]
        return UniversalDeckEncoding(
            global_hidden=global_hidden,
            option_hidden=option_hidden,
            option_mask=option_mask.bool(),
            deck_hidden=x[:, deck_begin:opponent_index],
            value_logits=self.value_head(global_hidden).squeeze(-1),
            opponent_logits=self.opponent_deck_head(x[:, opponent_index]),
        )

    def decoder_logits(
        self,
        encoding: UniversalDeckEncoding,
        selected_mask: torch.Tensor,
        min_count: torch.Tensor,
        max_count: torch.Tensor,
    ) -> torch.Tensor:
        """Return masked logits for [options..., STOP]."""
        selected_mask = selected_mask.bool()
        selected_count = selected_mask.sum(dim=1)
        selected_sum = (encoding.option_hidden * selected_mask[:, :, None]).sum(dim=1)
        selected_mean = selected_sum / selected_count.clamp_min(1).to(selected_sum.dtype)[:, None]
        step = selected_count.clamp_max(self.config.max_actions)
        context = torch.tanh(
            self.decoder_context(
                torch.cat((encoding.global_hidden, selected_mean, self.decoder_step(step)), dim=1)
            )
        )
        candidate = (
            self.decoder_option(encoding.option_hidden) * context[:, None, :]
        ).sum(dim=2) * (self.config.d_model ** -0.5)
        candidate = candidate + self.decoder_option_bias(encoding.option_hidden).squeeze(-1)
        candidate_valid = encoding.option_mask & ~selected_mask & (selected_count < max_count)[:, None]
        candidate = candidate.masked_fill(~candidate_valid, -1e4)
        stop = self.decoder_stop(context).squeeze(-1)
        stop = stop.masked_fill(selected_count < min_count, -1e4)
        return torch.cat((candidate, stop[:, None]), dim=1)

    def forward(self, *args: torch.Tensor) -> UniversalDeckEncoding:
        return self.encode(*args)

    @torch.no_grad()
    def greedy_actions(
        self,
        encoding: UniversalDeckEncoding,
        min_count: torch.Tensor,
        max_count: torch.Tensor,
    ) -> list[list[int]]:
        batch, options = encoding.option_mask.shape
        selected = torch.zeros((batch, options), dtype=torch.bool, device=encoding.option_mask.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=encoding.option_mask.device)
        result: list[list[int]] = [[] for _ in range(batch)]
        for _ in range(options + 1):
            logits = self.decoder_logits(encoding, selected, min_count, max_count)
            choice = stable_torch_argmax(logits)
            for row in range(batch):
                if bool(finished[row]):
                    continue
                index = int(choice[row])
                if index == options:
                    finished[row] = True
                else:
                    selected[row, index] = True
                    result[row].append(index)
            if bool(finished.all()):
                break
        return result

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def joint_policy_loss(
    model: UniversalDeckTransformerPolicy,
    encoding: UniversalDeckEncoding,
    original_labels: torch.Tensor,
    min_count: torch.Tensor,
    max_count: torch.Tensor,
    policy_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Teacher-forced negative log likelihood of exact option sequence + STOP."""
    labels = original_labels.bool() & encoding.option_mask
    batch, options = labels.shape
    chosen_count = labels.sum(dim=1)
    selected = torch.zeros_like(labels)
    per_decision = torch.zeros(batch, dtype=encoding.global_hidden.dtype, device=labels.device)
    steps = torch.zeros_like(per_decision)
    max_steps = int(chosen_count.max().detach().cpu()) + 1
    for step_index in range(max_steps):
        active = chosen_count >= step_index
        logits = model.decoder_logits(encoding, selected, min_count, max_count)
        remaining = labels & ~selected
        # Replay actions are sets. Canonical index order makes teacher forcing deterministic.
        next_option = remaining.to(torch.int64).argmax(dim=1)
        stop_index = torch.full_like(next_option, options)
        target = torch.where(chosen_count > step_index, next_option, stop_index)
        loss = F.cross_entropy(logits, target, reduction="none")
        per_decision = per_decision + loss * active
        steps = steps + active
        choose_option = active & (target < options)
        next_selected = selected.clone()
        if bool(choose_option.any()):
            rows = torch.arange(batch, device=labels.device)[choose_option]
            next_selected[rows, target[choose_option]] = True
        selected = next_selected
    per_decision = per_decision / steps.clamp_min(1.0)
    denominator = policy_weights.sum().clamp_min(1e-6)
    loss = (per_decision * policy_weights).sum() / denominator
    return loss, {
        "policyNll": loss.detach(),
        "meanTeacherSteps": steps.mean().detach(),
        "policyExamples": (policy_weights > 0).sum().detach(),
    }


def universal_bc_loss(
    model: UniversalDeckTransformerPolicy,
    encoding: UniversalDeckEncoding,
    batch: dict[str, torch.Tensor],
    policy_weights: torch.Tensor,
    value_weights: torch.Tensor,
    *,
    value_loss_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    policy, parts = joint_policy_loss(
        model,
        encoding,
        batch["original_labels"],
        batch["min_count"],
        batch["max_count"],
        policy_weights,
    )
    value_per_decision = F.binary_cross_entropy_with_logits(
        encoding.value_logits, batch["winner"], reduction="none"
    )
    value = (value_per_decision * value_weights).sum() / value_weights.sum().clamp_min(1e-6)
    total = policy + value_loss_weight * value
    return total, {**parts, "value": value.detach(), "total": total.detach()}
