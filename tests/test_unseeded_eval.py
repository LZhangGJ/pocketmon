from __future__ import annotations

import ast
import socket
import sys
import tempfile
import json
import os
import types
import unittest
from pathlib import Path

from rl.unseeded_eval import (
    NetworkBlocker,
    OfficialCabtModuleFinder,
    alternating_schedule,
    approved_terminal,
    install_agent_cg_alias,
    outcome_from_rewards,
    require_sha256,
    run_game_subprocess,
    sha256_file,
    summarize_stage_a,
)


class UnseededEvaluationTests(unittest.TestCase):
    def test_agent_cg_alias_reuses_verified_sim_module(self) -> None:
        previous_cg = sys.modules.get("cg")
        previous_sim = sys.modules.get("cg.sim")
        try:
            with tempfile.TemporaryDirectory() as directory:
                cg = Path(directory)
                (cg / "__init__.py").write_text("", encoding="utf-8")
                verified_sim = types.ModuleType("verified_sim")
                package = install_agent_cg_alias(cg, verified_sim)
                self.assertEqual(package.__path__, [str(cg.resolve())])
                self.assertIs(sys.modules["cg.sim"], verified_sim)
        finally:
            if previous_cg is None:
                sys.modules.pop("cg", None)
            else:
                sys.modules["cg"] = previous_cg
            if previous_sim is None:
                sys.modules.pop("cg.sim", None)
            else:
                sys.modules["cg.sim"] = previous_sim

    def test_runner_has_no_json_boolean_identifiers(self) -> None:
        runner = Path(__file__).resolve().parents[1] / "scripts/evaluate_unseeded_runtime.py"
        tree = ast.parse(runner.read_text(encoding="utf-8"))
        invalid = sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id in {"true", "false", "null"}})
        self.assertEqual(invalid, [])

    def test_sha_verification_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            path.write_bytes(b"official")
            digest = sha256_file(path)
            self.assertEqual(require_sha256(path, digest, "artifact"), digest)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                require_sha256(path, "0" * 64, "artifact")

    def test_module_finder_only_redirects_game_and_sim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cg = Path(directory)
            (cg / "game.py").write_text("VALUE = 1\n", encoding="utf-8")
            (cg / "sim.py").write_text("VALUE = 2\n", encoding="utf-8")
            finder = OfficialCabtModuleFinder(cg)
            game_spec = finder.find_spec("kaggle_environments.envs.cabt.cg.game")
            sim_spec = finder.find_spec("kaggle_environments.envs.cabt.cg.sim")
            self.assertEqual(Path(game_spec.origin), cg / "game.py")
            self.assertEqual(Path(sim_spec.origin), cg / "sim.py")
            self.assertIsNone(finder.find_spec("cg.api"))

    def test_approved_terminal_contract(self) -> None:
        for rewards in ([1, -1], [-1, 1], [0, 0]):
            self.assertTrue(approved_terminal(["DONE", "DONE"], rewards, True))
        self.assertFalse(approved_terminal(["DONE", "INVALID"], [1, -1], True))
        self.assertFalse(approved_terminal(["DONE", "DONE"], [1, 0], True))
        self.assertFalse(approved_terminal(["DONE", "DONE"], [1, -1], False))

    def test_outcome_uses_rewards_only(self) -> None:
        self.assertEqual(outcome_from_rewards([1, -1]), "seat0_win")
        self.assertEqual(outcome_from_rewards([-1, 1]), "seat1_win")
        self.assertEqual(outcome_from_rewards([0, 0]), "draw")
        self.assertIsNone(outcome_from_rewards([1, 0]))

    def test_schedule_is_interleaved_balanced_and_unpaired(self) -> None:
        schedule = alternating_schedule(20)
        self.assertEqual([row["game_id"] for row in schedule], list(range(1, 21)))
        self.assertEqual(sum(row["seat0"] == "official_random_first" for row in schedule), 10)
        self.assertEqual(sum(row["seat1"] == "official_random_first" for row in schedule), 10)
        self.assertTrue(all("seed" not in row and "pair" not in row for row in schedule))

    def test_network_blocker_counts_and_restores(self) -> None:
        original = socket.create_connection
        blocker = NetworkBlocker()
        with blocker:
            with self.assertRaisesRegex(RuntimeError, "network access disabled"):
                socket.create_connection(("example.com", 80))
        self.assertEqual(blocker.attempts, 1)
        self.assertIs(socket.create_connection, original)

    def test_summary_retains_failures_and_requires_all_games(self) -> None:
        records = [
            {
                "game_id": 1,
                "normal_terminal": True,
                "process_crash": False,
                "statuses": ["DONE", "DONE"],
                "native_hash_verified": True,
            },
            {
                "game_id": 2,
                "normal_terminal": False,
                "process_crash": False,
                "statuses": ["INVALID", "DONE"],
                "native_hash_verified": True,
            },
        ]
        summary = summarize_stage_a(records, expected_games=2, network_attempts=0)
        self.assertEqual(summary["games"], 2)
        self.assertEqual(summary["invalid_actions"], 1)
        self.assertEqual(summary["exceptions"], 0)
        self.assertFalse(summary["gate_passed"])

    @unittest.skipUnless(os.name == "posix", "POSIX signal return codes are Linux-specific")
    def test_subprocess_records_signal_without_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "missing.json"
            row = run_game_subprocess(
                [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"],
                result,
                5,
            )
        self.assertTrue(row["process_crash"])
        self.assertEqual(row["child_signal"], 15)
        self.assertEqual(row["retry_count"], 0)

    def test_subprocess_hard_timeout_retains_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            row = run_game_subprocess(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                Path(directory) / "missing.json",
                0.05,
            )
        self.assertTrue(row["hard_timeout"])
        self.assertFalse(row["normal_terminal"])
        self.assertEqual(row["retry_count"], 0)

    def test_subprocess_preserves_partial_child_evidence_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "partial.json"
            code = (
                "import json,sys; "
                f"open({str(result)!r},'w').write(json.dumps({{'game_id': 7, 'phase': 'loaded'}})); "
                "sys.exit(3)"
            )
            row = run_game_subprocess([sys.executable, "-c", code], result, 5)
        self.assertEqual(row["game_id"], 7)
        self.assertEqual(row["phase"], "loaded")
        self.assertEqual(row["child_exit_code"], 3)
        self.assertFalse(row["process_crash"])
        self.assertTrue(row["abnormal_exit"])
        self.assertEqual(row["retry_count"], 0)


if __name__ == "__main__":
    unittest.main()
