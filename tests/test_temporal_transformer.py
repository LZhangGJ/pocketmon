from __future__ import annotations

import torch

from rl.bc import collate_rows
from rl.features import ACTION_DIM, HISTORY_DIM, STATE_DIM
from rl.model import StructuredMaskedPointerActorCritic
from rl.temporal_model import (
    StructuredTemporalTransformerActorCritic,
    load_structured_warm_start,
)


def make_row(history_steps: int) -> dict:
    return {
        "state": [0.0] * STATE_DIM,
        "options": [
            [0.0] * ACTION_DIM,
            [0.1] * ACTION_DIM,
            [0.2] * ACTION_DIM,
        ],
        "action": [1],
        "history": [
            [float(step + 1) / 10.0] * HISTORY_DIM
            for step in range(history_steps)
        ],
        "min_count": 1,
        "max_count": 1,
        "outcome": 1.0,
        "policy_weight": 1.0,
        "value_weight": 1.0,
        "option_card_ids": [1, 2, 3],
        "option_target_card_ids": [0, 4, 5],
        "option_attack_ids": [0, 6, 7],
        "entity_card_ids": [1, 8],
        "entity_zone_ids": [1, 5],
        "entity_features": [
            [0.0] * 8,
            [0.25] * 8,
        ],
        "deck_card_ids": [1] * 60,
    }


def test_temporal_transformer_handles_empty_and_nonempty_histories() -> None:
    batch = collate_rows([make_row(0), make_row(3)])
    model = StructuredTemporalTransformerActorCritic(
        hidden_dim=32,
        history_length=8,
        transformer_layers=1,
        transformer_heads=4,
        transformer_ffn_dim=64,
        transformer_dropout=0.0,
    )
    model.eval()
    with torch.inference_mode():
        state, options, values = model.encode_batch(batch)
    assert state.shape == (2, 32)
    assert options.shape == (2, 3, 32)
    assert values.shape == (2,)
    assert torch.isfinite(state).all()
    assert torch.isfinite(options).all()
    assert torch.isfinite(values).all()
    assert not torch.allclose(state[0], state[1])


def test_structured_checkpoint_warm_start_requires_all_base_weights() -> None:
    base = StructuredMaskedPointerActorCritic(hidden_dim=32)
    temporal = StructuredTemporalTransformerActorCritic(
        hidden_dim=32,
        history_length=8,
        transformer_layers=1,
        transformer_heads=4,
        transformer_ffn_dim=64,
        transformer_dropout=0.0,
    )
    result = load_structured_warm_start(temporal, base.state_dict())
    assert result["unexpected_keys"] == []
    assert result["missing_keys"]
    assert all(
        key.startswith(
            (
                "history_input.",
                "history_positions.",
                "history_encoder.",
                "history_output.",
                "history_gate.",
                "temporal_context_norm.",
            )
        )
        for key in result["missing_keys"]
    )
