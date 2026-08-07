from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from unittest.mock import patch

from scripts.continuous_rl_pipeline import (
    build_rollout_pool,
    choose_trainer,
    generation_training_config,
    retain_candidate_in_league,
    wait_for_files,
)


class ContinuousPipelineTests(unittest.TestCase):
    def test_rollout_pool_marks_public_and_frozen_population_roles(self) -> None:
        pool = build_rollout_pool(
            [{"name": "recent", "agent_dir": "/agents/recent"}],
            ["/agents/champion", "/agents/history"],
        )
        self.assertEqual([item["league_role"] for item in pool], ["public", "population", "population"])
        self.assertEqual(pool[1]["name"], "population_00")

    def test_rollout_pool_rejects_missing_frozen_population(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one population"):
            build_rollout_pool([], [])

    def test_pbt_variants_rotate_deterministically(self) -> None:
        config = {
            "ppo_learning_rate": 1e-5,
            "entropy_coefficient": 0.01,
            "ppo_epochs": 2,
            "frozen_league_fraction": 0.6,
            "rollout_temperature": 1.0,
            "gamma": 0.997,
            "gae_lambda": 0.95,
            "clip_ratio": 0.1,
            "value_clip": 0.2,
            "value_coefficient": 0.5,
            "gradient_clip_norm": 0.5,
            "target_kl": 0.03,
            "pbt_variants": [
                {"name": "base"},
                {"name": "explore", "multipliers": {"ppo_learning_rate": 1.5},
                 "overrides": {"rollout_temperature": 1.1}},
            ],
        }
        first, first_record = generation_training_config(config, 1)
        second, second_record = generation_training_config(config, 2)
        third, third_record = generation_training_config(config, 3)
        self.assertEqual((first_record["name"], second_record["name"], third_record["name"]),
                         ("base", "explore", "base"))
        self.assertEqual(first["ppo_learning_rate"], 1e-5)
        self.assertAlmostEqual(second["ppo_learning_rate"], 1.5e-5)
        self.assertEqual(second["rollout_temperature"], 1.1)
        self.assertEqual(third["rollout_temperature"], 1.0)

    def test_rejected_safe_candidate_can_join_training_league(self) -> None:
        config = {
            "retain_rejected_in_league": True,
            "league_retention_min_public_score": 0.2,
            "league_retention_max_worst_regression": 0.75,
        }
        report = {
            "checks": {"zero_failures": True},
            "candidate_public": {"score_rate": 0.3},
            "worst_matchup_delta": -0.5,
        }
        self.assertTrue(retain_candidate_in_league(config, report))
        report["checks"]["zero_failures"] = False
        self.assertFalse(retain_candidate_in_league(config, report))

    def test_waits_for_delayed_shared_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "remote-result.json"

            def create() -> None:
                time.sleep(0.05)
                path.write_text("{}", encoding="utf-8")

            worker = threading.Thread(target=create)
            worker.start()
            wait_for_files([path], timeout_seconds=2.0)
            worker.join()
            self.assertTrue(path.is_file())

    def test_reports_missing_shared_artifact_after_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing"
            with self.assertRaisesRegex(RuntimeError, "did not become visible"):
                wait_for_files([path], timeout_seconds=0.01)

    def test_trainer_selection_prefers_idle_gpu_over_larger_busy_gpu(self) -> None:
        readings = {
            "idle": (20_000, 24_000, 0, 1),
            "busy": (70_000, 80_000, 92, 3),
        }
        with patch("scripts.continuous_rl_pipeline.torch_cuda_device_count", return_value=4), patch(
            "scripts.continuous_rl_pipeline.free_gpu", side_effect=lambda host, _: readings[host]
        ):
            host, gpu, audit = choose_trainer({
                "trainer_hosts": ["busy", "idle"],
                "local_host": "coordinator",
            })
        self.assertEqual((host, gpu), ("idle", 1))
        self.assertEqual(audit["busy"]["utilization_percent"], 92)

    def test_trainer_selection_skips_driver_incompatible_host(self) -> None:
        with patch(
            "scripts.continuous_rl_pipeline.torch_cuda_device_count",
            side_effect=lambda host, _: {"broken": 0, "usable": 2}[host],
        ), patch("scripts.continuous_rl_pipeline.free_gpu", return_value=(20_000, 24_000, 0, 1)):
            host, gpu, audit = choose_trainer({
                "trainer_hosts": ["broken", "usable"],
                "local_host": "coordinator",
            })
        self.assertEqual((host, gpu), ("usable", 1))
        self.assertIn("cannot initialize CUDA", audit["broken"]["error"])


if __name__ == "__main__":
    unittest.main()
