from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.async_gold_v31_pipeline import (
    choose_candidate_kind,
    count_pending,
    deep_merge,
    load_config,
    package_fingerprint,
    promote_compare_and_swap,
    select_candidate,
)
from scripts.continuous_rl_pipeline import atomic_json, ensure_action_q_checkpoint


class AsyncGoldV31Tests(unittest.TestCase):
    def test_deep_merge_preserves_base_and_replaces_nested_values(self) -> None:
        merged = deep_merge(
            {"a": 1, "nested": {"x": 2, "y": 3}},
            {"nested": {"x": 7}, "b": 4},
        )
        self.assertEqual(merged, {"a": 1, "nested": {"x": 7, "y": 3}, "b": 4})

    def test_shipped_configs_use_exact_stages_and_dueling_q(self) -> None:
        for name in ("async_gold_v31_garchomp.json", "async_gold_v31_grimmsnarl.json"):
            config, async_config = load_config(Path("configs") / name)
            self.assertEqual([stage["target_games"] for stage in config["gate_stages"]], [20, 200, 400])
            self.assertTrue(config["action_q_dueling_advantage"])
            self.assertEqual(config["action_q_loss_priority_weight"], 4.0)
            self.assertGreaterEqual(async_config["max_pending_candidates"], 4)

    def test_deck_candidates_are_periodic_not_every_generation(self) -> None:
        settings = {"deck_candidate_every": 4}
        self.assertEqual([choose_candidate_kind(i, settings) for i in range(1, 6)], [
            "policy", "policy", "policy", "deck", "policy",
        ])

    def test_queue_resumes_evaluating_then_prefers_current_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = [
                (1, "queued", 0, 100),
                (2, "queued", 1, 90),
                (3, "evaluating", 0, 50),
            ]
            for generation, status, parent_version, priority in candidates:
                candidate_root = root / f"generation_{generation:05d}"
                atomic_json(candidate_root / "candidate.json", {
                    "generation": generation,
                    "training_parent_version": parent_version,
                    "priority": priority,
                    "queued_at": f"2026-08-08T00:00:0{generation}+00:00",
                })
                atomic_json(candidate_root / "lifecycle.json", {"status": status})
            self.assertEqual(count_pending(root), 3)
            selected = select_candidate(root, champion_version=1)
            self.assertIsNotNone(selected)
            self.assertEqual(selected[1]["generation"], 3)
            atomic_json(root / "generation_00003" / "lifecycle.json", {"status": "rejected"})
            selected = select_candidate(root, champion_version=1)
            self.assertEqual(selected[1]["generation"], 2)

    def test_package_fingerprint_changes_with_deck(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = []
            for index, deck_value in enumerate(("1\n" * 60, "2\n" * 60)):
                package = root / str(index)
                package.mkdir()
                (package / "checkpoint.pt").write_bytes(b"same-model")
                (package / "deck.csv").write_text(deck_value, encoding="utf-8")
                values.append(package_fingerprint(package)["candidate_sha256"])
            self.assertNotEqual(*values)

    def test_champion_update_is_compare_and_swap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "champion.json", {
                "version": 2, "package": "old", "checkpoint_sha256": "old-sha",
            })
            manifest = {
                "generation": 7,
                "package": "new",
                "checkpoint_sha256": "new-sha",
                "deck_sha256": "deck-sha",
                "action_q_sha256": None,
                "candidate_sha256": "candidate-sha",
            }
            self.assertFalse(promote_compare_and_swap(root, manifest, {}, expected_version=1))
            self.assertTrue(promote_compare_and_swap(root, manifest, {"promotion_report": "report"}, expected_version=2))
            self.assertEqual((root / "champion.json").read_text(encoding="utf-8").count('"version": 3'), 1)

    def test_dueling_flag_reaches_action_q_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rollouts").mkdir()
            (root / "rollouts" / "shard_000.counterfactual.jsonl.gz").write_bytes(b"x")
            champion = root / "champion"
            champion.mkdir()
            captured = {}

            def fake_run_process(**kwargs):
                captured["command"] = kwargs["command"]

            config = {
                "counterfactual_rate": 0.02,
                "python": "python",
                "code_root": str(root),
                "action_q_epochs": 8,
                "action_q_batch_size": 32,
                "action_q_learning_rate": 1e-4,
                "action_q_heads": 3,
                "action_q_dueling_advantage": True,
                "base_seed": 10,
                "local_host": "local",
                "host_environment": {},
            }
            paths = {"rollouts": root / "rollouts", "q_train": root / "q"}
            with patch("scripts.continuous_rl_pipeline.choose_trainer", return_value=("local", 0, {})), patch(
                "scripts.continuous_rl_pipeline.run_process", side_effect=fake_run_process,
            ), patch("scripts.continuous_rl_pipeline.wait_for_files"):
                ensure_action_q_checkpoint(
                    config=config,
                    state={"champion_package": str(champion)},
                    generation=1,
                    paths=paths,
                    actor_checkpoint=root / "actor.pt",
                    events=[],
                )
            self.assertIn("--dueling-advantage", captured["command"])


if __name__ == "__main__":
    unittest.main()
