from __future__ import annotations

import random
import unittest

import torch

from rl.action_q import ActionValueEnsemble, DuelingActionValueEnsemble, q_mean_and_std
from rl.agent_adapter import apply_q_override_budget, conservative_q_choice
from rl.counterfactual import sample_hidden_zones, terminal_value


class ActionQTests(unittest.TestCase):
    def test_q_only_overrides_with_margin_and_low_uncertainty(self):
        candidates = torch.tensor([0, 1, 2])
        chosen, status, _, _ = conservative_q_choice(
            0, candidates, torch.tensor([0.1, 0.4, 0.2]), torch.tensor([0.05, 0.08, 0.04]),
            min_margin=0.15, max_uncertainty=0.10,
        )
        self.assertEqual((chosen, status), (1, "override"))

        chosen, status, _, _ = conservative_q_choice(
            0, candidates, torch.tensor([0.1, 0.2, 0.0]), torch.tensor([0.05, 0.08, 0.04]),
            min_margin=0.15, max_uncertainty=0.10,
        )
        self.assertEqual((chosen, status), (0, "abstain"))

        chosen, status, _, _ = conservative_q_choice(
            0, candidates, torch.tensor([0.1, 0.4, 0.2]), torch.tensor([0.05, 0.20, 0.04]),
            min_margin=0.15, max_uncertainty=0.10,
        )
        self.assertEqual((chosen, status), (0, "abstain"))

    def test_action_q_shapes_and_finite_uncertainty(self) -> None:
        model = ActionValueEnsemble(hidden_dim=8, heads=3)
        values = model(torch.randn(2, 8), torch.randn(2, 5, 8))
        mean, std = q_mean_and_std(values)
        self.assertEqual(tuple(values.shape), (2, 5, 3))
        self.assertEqual(tuple(mean.shape), (2, 5))
        self.assertTrue(torch.isfinite(std).all())

    def test_dueling_q_exposes_action_advantage(self) -> None:
        model = DuelingActionValueEnsemble(hidden_dim=8, heads=5)
        q, advantage = model.q_and_advantage(torch.randn(2, 8), torch.randn(2, 4, 8))
        self.assertEqual(tuple(q.shape), (2, 4, 5))
        self.assertEqual(tuple(advantage.shape), (2, 4, 5))
        self.assertTrue(torch.isfinite(q).all())

    def test_q_override_budget_caps_takeover_rate(self) -> None:
        credit = 0.0
        statuses = []
        choices = []
        for _ in range(10):
            choice, status, credit = apply_q_override_budget(0, 1, "override", credit, 0.20)
            choices.append(choice)
            statuses.append(status)
        self.assertEqual(choices.count(1), 2)
        self.assertEqual(statuses.count("override"), 2)
        self.assertEqual(statuses.count("budget_abstain"), 8)

    def test_exact_training_decks_fill_hidden_zones(self) -> None:
        decks = [[1] * 30 + [2] * 30, [3] * 30 + [4] * 30]
        observation = {
            "current": {
                "yourIndex": 0,
                "result": -1,
                "players": [
                    {"hand": [{"id": 1}], "discard": [], "active": [{"id": 2}], "bench": [], "prize": [None] * 6, "deckCount": 52, "handCount": 1},
                    {"hand": [], "discard": [{"id": 3}], "active": [None], "bench": [], "prize": [None] * 6, "deckCount": 46, "handCount": 7},
                ],
            }
        }
        hidden = sample_hidden_zones(observation, decks, {2, 4}, random.Random(7))
        self.assertEqual(len(hidden["your_deck"]), 52)
        self.assertEqual(len(hidden["your_prize"]), 6)
        self.assertEqual(len(hidden["opponent_deck"]), 46)
        self.assertEqual(len(hidden["opponent_hand"]), 7)
        self.assertEqual(hidden["opponent_active"], [4])

    def test_terminal_value_uses_root_player(self) -> None:
        self.assertEqual(terminal_value({"current": {"result": 1}}, 1), 1.0)
        self.assertEqual(terminal_value({"current": {"result": 0}}, 1), -1.0)
        self.assertIsNone(terminal_value({"current": {"result": -1}}, 1))


if __name__ == "__main__":
    unittest.main()
