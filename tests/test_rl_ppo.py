from __future__ import annotations

import unittest

import torch

from rl.bc import action_is_legal, collate_rows
from rl.model import StructuredMaskedPointerActorCritic
from rl.ppo import (
    compute_gae,
    evaluate_action_sequences,
    model_row_from_observation,
    normalize_advantages,
    ppo_batch_loss,
    sample_action,
)


def observation() -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "turn": 2,
            "players": [
                {"active": [{"id": 10, "hp": 100, "maxHp": 120}], "bench": [], "hand": [{"id": 11}], "discard": []},
                {"active": [{"id": 20, "hp": 80, "maxHp": 100}], "bench": [], "hand": [None], "discard": []},
            ],
        },
        "select": {
            "type": 1,
            "minCount": 1,
            "maxCount": 2,
            "option": [{"type": 1}, {"type": 2}, {"type": 3}],
        },
    }


class MaskedPPOTests(unittest.TestCase):
    def test_sample_and_recompute_joint_log_probability(self) -> None:
        torch.manual_seed(7)
        model = StructuredMaskedPointerActorCritic(32).eval()
        deck = [10, 11, 20] * 20
        action, behavior_log_probability, value, entropy = sample_action(
            model, observation(), deck, torch.device("cpu")
        )
        self.assertTrue(action_is_legal(action, 3, 1, 2))
        row = model_row_from_observation(observation(), deck, action)
        batch = collate_rows([row])
        recomputed, recomputed_entropy, recomputed_value = evaluate_action_sequences(model, batch)
        self.assertAlmostEqual(float(recomputed[0].detach()), behavior_log_probability, places=5)
        self.assertAlmostEqual(float(recomputed_value[0].detach()), value, places=5)
        self.assertAlmostEqual(float(recomputed_entropy[0].detach()), entropy, places=5)

    def test_gae_is_computed_per_player_sequence(self) -> None:
        rows = [
            {"episode_id": "e", "player": 0, "action_step": 1, "behavior_value": 0.1, "reward": 0.0},
            {"episode_id": "e", "player": 1, "action_step": 2, "behavior_value": -0.3, "reward": -1.0},
            {"episode_id": "e", "player": 0, "action_step": 3, "behavior_value": 0.2, "reward": 1.0},
        ]
        result = compute_gae(rows, gamma=0.9, gae_lambda=0.8)
        player_zero = [row for row in result if row["player"] == 0]
        self.assertAlmostEqual(player_zero[1]["advantage"], 0.8)
        expected_first = (0.9 * 0.2 - 0.1) + 0.9 * 0.8 * 0.8
        self.assertAlmostEqual(player_zero[0]["advantage"], expected_first)
        self.assertAlmostEqual([row for row in result if row["player"] == 1][0]["advantage"], -0.7)

    def test_ppo_loss_is_finite_and_backpropagates(self) -> None:
        torch.manual_seed(11)
        model = StructuredMaskedPointerActorCritic(32)
        deck = [10, 11, 20] * 20
        rows = []
        for index, action in enumerate(([0], [1, 2])):
            row = model_row_from_observation(observation(), deck, list(action))
            batch = collate_rows([row])
            with torch.no_grad():
                log_probability, _, value = evaluate_action_sequences(model, batch)
            row.update({
                "episode_id": f"e{index}",
                "player": 0,
                "action_step": 1,
                "behavior_log_probability": float(log_probability[0]),
                "behavior_value": float(value[0]),
                "reward": 1.0 if index == 0 else -1.0,
            })
            rows.append(row)
        prepared = normalize_advantages(compute_gae(rows))
        loss, metrics = ppo_batch_loss(model, collate_rows(prepared))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertAlmostEqual(metrics["ratio_mean"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
