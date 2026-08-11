from __future__ import annotations

import os
import sys
import tempfile
import unittest
import gzip
import json
from pathlib import Path

import numpy as np
import torch


INTEGRATION = Path(__file__).resolve().parent
REFERENCE = INTEGRATION.parent / "reference"
for path in (INTEGRATION, REFERENCE / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from universal_deck_model import UniversalDeckModelConfig, UniversalDeckTransformerPolicy  # noqa: E402
from universal_ppo import (  # noqa: E402
    collate_rows,
    compute_gae,
    evaluate_actions,
    normalize_advantages,
    ppo_loss,
    sample_action,
)
from train_universal_ppo import load_rollouts, require_clean_repository  # noqa: E402
from collect_universal_ppo_rollouts import canonical_archetype  # noqa: E402
from common import Experiment7Error  # noqa: E402
from universal_ppo import ROLLOUT_FORMAT  # noqa: E402


def row(action: list[int] | None = None) -> dict:
    config = UniversalDeckModelConfig(d_model=32, n_heads=4, n_layers=1, ff_dim=64)
    return {
        "state": np.zeros(config.state_dim, dtype=np.float32).tolist(),
        "history_state": np.zeros((config.history_length, config.state_dim), dtype=np.float32).tolist(),
        "history_action": np.zeros((config.history_length, config.option_dim), dtype=np.float32).tolist(),
        "history_mask": np.zeros(config.history_length, dtype=np.uint8).tolist(),
        "own_deck_cards": ([7, 104, 112] * 20),
        "entity_cat": np.zeros((4, 10), dtype=np.int64).tolist(),
        "entity_num": np.zeros((4, config.entity_num_dim), dtype=np.float32).tolist(),
        "entity_mask": [1, 1, 0, 0],
        "options": np.zeros((3, config.option_dim), dtype=np.float32).tolist(),
        "min_count": 1,
        "max_count": 2,
        "action": list(action or []),
    }


class UniversalPpoTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        config = UniversalDeckModelConfig(d_model=32, n_heads=4, n_layers=1, ff_dim=64)
        self.model = UniversalDeckTransformerPolicy(config)
        self.device = torch.device("cpu")

    def test_sample_and_recompute_log_probability(self) -> None:
        action, log_probability, value, entropy = sample_action(
            self.model.eval(), row(), self.device
        )
        prepared = row(action)
        recomputed, recomputed_entropy, recomputed_value = evaluate_actions(
            self.model, collate_rows([prepared], self.device)
        )
        self.assertAlmostEqual(float(recomputed[0]), log_probability, places=5)
        self.assertAlmostEqual(float(recomputed_value[0]), value, places=5)
        self.assertAlmostEqual(float(recomputed_entropy[0]), entropy, places=5)

    def test_ppo_loss_backpropagates(self) -> None:
        rows = []
        for index in range(2):
            action, log_probability, value, entropy = sample_action(
                self.model.eval(), row(), self.device
            )
            item = row(action)
            item.update(
                {
                    "episode_id": f"episode-{index}",
                    "player": 0,
                    "action_step": 1,
                    "behavior_log_probability": log_probability,
                    "teacher_log_probability": log_probability,
                    "behavior_value": value,
                    "behavior_entropy": entropy,
                    "reward": 1.0 if index == 0 else -1.0,
                }
            )
            rows.append(item)
        prepared = normalize_advantages(compute_gae(rows))
        self.model.train()
        loss, metrics = ppo_loss(self.model, collate_rows(prepared, self.device))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))
        self.assertAlmostEqual(metrics["ratioMean"], 1.0, places=5)

    def test_ppo_loss_can_upweight_second_seat(self) -> None:
        action, log_probability, value, entropy = sample_action(
            self.model.eval(), row(), self.device
        )
        rows = []
        for player, advantage in ((0, 1.0), (1, -1.0)):
            item = row(action)
            item.update(
                {
                    "episode_id": f"seat-{player}",
                    "player": player,
                    "action_step": 1,
                    "behavior_log_probability": log_probability,
                    "teacher_log_probability": log_probability,
                    "behavior_value": value,
                    "behavior_entropy": entropy,
                    "advantage": advantage,
                    "return": value,
                }
            )
            rows.append(item)
        _, balanced = ppo_loss(self.model, collate_rows(rows, self.device))
        _, weighted = ppo_loss(
            self.model, collate_rows(rows, self.device), seat1_weight=2.0
        )
        self.assertAlmostEqual(balanced["policyLoss"], 0.0, places=5)
        self.assertAlmostEqual(weighted["policyLoss"], 1.0 / 3.0, places=5)
        self.assertEqual(weighted["seat1Weight"], 2.0)

    def test_repository_validation_does_not_depend_on_process_cwd(self) -> None:
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.chdir(temporary)
                repository, commit = require_clean_repository(INTEGRATION / "train_universal_ppo.py")
        finally:
            os.chdir(previous)
        self.assertTrue((repository / ".git").exists() or (repository / ".git").is_file())
        self.assertEqual(len(commit), 40)

    def test_canonical_archetype_collapses_b08_aliases(self) -> None:
        self.assertEqual(canonical_archetype({"archetype": "alakazam"}), "A03")
        self.assertEqual(canonical_archetype({"archetype": "dragapult"}), "A06")
        self.assertEqual(canonical_archetype({"archetype": "A08"}), "A08")
        self.assertEqual(
            canonical_archetype({"archetype": "A03", "canonical_archetype": "A07"}),
            "A07",
        )

    def test_async_rollouts_accept_bounded_staleness(self) -> None:
        item = row([0])
        item.update(
            {
                "rollout_format": ROLLOUT_FORMAT,
                "episode_id": "async-episode",
                "player": 0,
                "action_step": 1,
                "behavior_checkpoint_sha256": "old-sha",
                "behavior_generation": 8,
                "teacher_checkpoint_sha256": "teacher-sha",
                "behavior_log_probability": 0.0,
                "teacher_log_probability": 0.0,
                "behavior_value": 0.0,
                "behavior_entropy": 0.0,
                "reward": 1.0,
                "outcome": 1.0,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rollout.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(item) + "\n")
            rows, _ = load_rollouts(
                [path],
                "current-sha",
                "teacher-sha",
                allowed_behavior_generations={"old-sha": 8},
                current_generation=10,
                max_behavior_lag=2,
            )
            self.assertEqual(len(rows), 1)
            with self.assertRaises(Experiment7Error):
                load_rollouts(
                    [path],
                    "current-sha",
                    "teacher-sha",
                    allowed_behavior_generations={"old-sha": 8},
                    current_generation=11,
                    max_behavior_lag=2,
                )


if __name__ == "__main__":
    unittest.main()
