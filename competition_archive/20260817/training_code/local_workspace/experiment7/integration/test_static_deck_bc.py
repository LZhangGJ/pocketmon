from __future__ import annotations

import json
import importlib
import importlib.util
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

try:
    from static_deck_bc_common import (
        STRICT_PREDICATE,
        assert_specialist_receipt,
        matching_archetypes,
        strict_row,
        validation_episode,
    )
except ModuleNotFoundError:
    from .static_deck_bc_common import (
        STRICT_PREDICATE,
        assert_specialist_receipt,
        matching_archetypes,
        strict_row,
        validation_episode,
    )


ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (ROOT / "experiment7/config/static_deck_bc_10d_20260815.json").read_text(encoding="utf-8")
)


class StaticDeckBcTest(unittest.TestCase):
    def test_config_fixes_ten_epoch_patience_one(self) -> None:
        training = CONFIG["training"]
        self.assertEqual(training["minEpochs"], 1)
        self.assertEqual(training["maxEpochs"], 10)
        self.assertEqual(training["patience"], 1)
        self.assertTrue(training["asynchronousValidation"])
        self.assertEqual(training["selectionMetric"], "exactSemantic")

    def test_strict_boundary_rejects_exactly_1000_and_unclean(self) -> None:
        self.assertTrue(strict_row({"is_clean": "1", "min_score": "1000.00001"}))
        self.assertFalse(strict_row({"is_clean": "1", "min_score": "1000"}))
        self.assertFalse(strict_row({"is_clean": "0", "min_score": "2000"}))

    def test_composite_archetype_beats_ogerpon_only(self) -> None:
        cards = ["Raging Bolt ex", "Teal Mask Ogerpon ex", "Team Rocket's Kangaskhan ex"]
        self.assertEqual(matching_archetypes(cards, CONFIG), ["raging_bolt_ogerpon_kangaskhan"])
        self.assertEqual(matching_archetypes(["Teal Mask Ogerpon ex"], CONFIG), ["ogerpon_only"])

    def test_loader_receipt_asserts_strict_score(self) -> None:
        receipt = {
            "strictPredicate": STRICT_PREDICATE,
            "minScoreExclusive": 1000.0,
            "episodes": 2,
            "scoreMin": 1000.1,
            "scoreMax": 1100.0,
            "duplicateEpisodes": 0,
            "trainValidationOverlap": 0,
        }
        assert_specialist_receipt(receipt)
        receipt["scoreMin"] = 1000.0
        with self.assertRaisesRegex(ValueError, "score <= 1000"):
            assert_specialist_receipt(receipt)

    def test_episode_split_is_stable(self) -> None:
        first = validation_episode("92473297", CONFIG)
        self.assertEqual(first, validation_episode("92473297", CONFIG))
        self.assertIsInstance(first, bool)

    def test_windows_root_and_cache_drive_are_d_only(self) -> None:
        windows = CONFIG["windows"]
        self.assertTrue(windows["root"].startswith("D:\\"))
        self.assertTrue(windows["persistentRoot"].startswith("D:\\"))
        self.assertTrue(windows["scratchRoot"].startswith("G:\\"))
        self.assertEqual(windows["finalArtifactCopies"], 2)
        self.assertNotIn("C:", windows["largeFilesAllowedDrives"])

    def test_static_sharder_imports(self) -> None:
        path = ROOT / "experiment7/integration/build_static_deck_day_shard.py"
        spec = importlib.util.spec_from_file_location("static_sharder_test", path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)

    def test_static_trainer_and_validator_compile_contract(self) -> None:
        for name in ("train_static_deck_bc_async.py", "validate_static_deck_bc_async.py"):
            path = ROOT / "experiment7/integration" / name
            self.assertTrue(path.is_file())
            text = path.read_text(encoding="utf-8")
            self.assertIn("validate_static_sources", text)
        validator = (ROOT / "experiment7/integration/validate_static_deck_bc_async.py").read_text(encoding="utf-8")
        self.assertIn("score > best_score", validator)

    def test_profile_prefers_local_frozen_initializer(self) -> None:
        text = (ROOT / "experiment7/integration/run_static_deck_bc_profile.py").read_text(encoding="utf-8")
        self.assertIn('local_root / "initializer.pt"', text)
        self.assertIn('"--initialize-from"', text)

    def test_smoke_is_one_batch_and_not_epoch(self) -> None:
        path = ROOT / "experiment7/integration/smoke_static_deck_bc_batch.py"
        text = path.read_text(encoding="utf-8")
        self.assertIn('"formalEpochStarted": False', text)
        self.assertNotIn("train_epoch(", text)

    def test_post_strict_controller_requires_success_and_ten_days(self) -> None:
        text = (ROOT / "experiment7/integration/static_deck_bc_post_strict_controller.py").read_text(encoding="utf-8")
        self.assertIn('root / "SUCCESS"', text)
        self.assertIn("len(datasets) != expected_days", text)
        self.assertIn('"formalTrainingStarted": False', text)
        self.assertIn("strict_verified_launching", text)

    def test_remote_local_authority_binds_manifest_sha_and_parity(self) -> None:
        path = ROOT / "experiment7/integration/static_deck_bc_post_strict_controller.py"
        spec = importlib.util.spec_from_file_location("static_post_strict_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        manifest_sha = "b80edd05263002dbaf830dc4ad0296b7ede435f7a861461a9f28838a72d5010b"
        with tempfile.TemporaryDirectory() as directory:
            parity = Path(directory) / "parity.json"
            parity.write_text(json.dumps({
                "status": "passed",
                "days": [{"name": f"day-{index}"} for index in range(10)],
            }), encoding="utf-8")
            completed = mock.Mock(returncode=0, stdout=f"{manifest_sha}  manifest\n")
            with mock.patch.object(module.subprocess, "run", return_value=completed):
                result = module.verify_remote_local_authority(
                    "doraemon17", Path("/tmp/strict"), 10, manifest_sha, parity,
                )
            self.assertTrue(result["localAuthority"])
            self.assertTrue(result["parity"])
            self.assertEqual(result["manifestSha256"], manifest_sha)
            self.assertEqual(len(result["days"]), 10)

    def test_cluster_plan_excludes_d16_and_reserves_windows(self) -> None:
        text = (ROOT / "experiment7/integration/launch_static_deck_bc_cluster.sh").read_text(encoding="utf-8")
        self.assertNotIn("doraemon16", text)
        self.assertIn("grimmsnarl_froslass_munkidori", text)
        self.assertIn("windows-rtx5070ti", text)
        self.assertEqual(text.count("run_static_deck_bc_profile.py"), 1)

    def test_lpt_plan_has_exactly_eleven_unique_profiles(self) -> None:
        plan = json.loads((ROOT / "experiment7/config/static_deck_bc_lpt_plan_20260815.json").read_text(encoding="utf-8"))
        names = [row["archetype"] for row in plan["assignments"]]
        self.assertEqual(len(names), 11)
        self.assertEqual(len(set(names)), 11)
        self.assertNotIn("doraemon16", [row["host"] for row in plan["assignments"]])
        self.assertFalse(plan["formalTrainingStarted"])

    def test_windows_watcher_waits_for_remote_ten_day_parity(self) -> None:
        text = (ROOT / "experiment7/integration/static_deck_bc_windows_post_strict_watcher.py").read_text(encoding="utf-8")
        self.assertIn('len(verification.get("days", [])) == 10', text)
        self.assertIn('"formalTrainingStarted": False', text)
        launcher = (ROOT / "experiment7/integration/windows_static_deck_bc_launch.py").read_text(encoding="utf-8")
        self.assertIn("scp", launcher)
        self.assertIn("local_sources = sources_manifest", launcher)
        self.assertIn("trainerPid", launcher)
        self.assertIn("frozenSha256", launcher)

    def test_windows_gpu_idle_accepts_wddm_baseline_only(self) -> None:
        path = ROOT / "experiment7/integration/windows_static_deck_bc_launch.py"
        spec = importlib.util.spec_from_file_location("windows_static_deck_bc_launch_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertTrue(module.gpu_sample_is_idle((2410, 10)))
        self.assertFalse(module.gpu_sample_is_idle((4096, 2)))
        self.assertFalse(module.gpu_sample_is_idle((2410, 30)))

    def test_static_pool_persistence_updates_base_and_live_idempotently(self) -> None:
        integration = str(ROOT / "experiment7/integration")
        sys.path.insert(0, integration)
        try:
            module = importlib.import_module("persist_static_deck_bc_pool")
        finally:
            sys.path.remove(integration)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            live = root / "live.json"
            league = root / "league.json"
            static = root / "static.json"
            receipt = root / "receipt.json"
            base.write_text(json.dumps({"agents": [{"name": "base"}]}), encoding="utf-8")
            live.write_text(json.dumps({"agents": []}), encoding="utf-8")
            league.write_text(json.dumps({
                "basePool": {"path": str(base)},
                "poolPath": str(live),
            }), encoding="utf-8")
            static_agent = {
                "name": "static-one",
                "sourceKind": "replay_static_bc",
                "immutable": True,
                "ppoUpdatesAllowed": False,
                "deck_canonical_sha256": "deck-a",
                "policyVersion": "policy-a",
            }
            static.write_text(json.dumps({"agents": [static_agent]}), encoding="utf-8")

            def build(payload):
                prospective = json.loads(Path(payload["basePool"]["path"]).read_text(encoding="utf-8"))
                return {"agents": prospective["agents"], "asyncLeague": {"basePool": {}}}

            with mock.patch.object(module, "build_pool_payload", side_effect=build):
                first = module.persist_static_pool(league, static, receipt)
                second = module.persist_static_pool(league, static, receipt)
            self.assertEqual(first["basePool"]["before"], 1)
            self.assertEqual(second["basePool"]["after"], 2)
            self.assertEqual(len(json.loads(base.read_text(encoding="utf-8"))["agents"]), 2)
            self.assertEqual(
                [row["name"] for row in json.loads(live.read_text(encoding="utf-8"))["agents"]],
                ["base", "static-one"],
            )


if __name__ == "__main__":
    unittest.main()
