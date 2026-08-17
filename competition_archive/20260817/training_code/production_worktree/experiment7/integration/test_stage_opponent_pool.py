from __future__ import annotations

import io
import json
import py_compile
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    Experiment7Error,
    canonical_deck_sha256,
    directory_sha256,
    read_deck,
    sha256_file,
    write_csv,
    write_json,
)
from arena import make_schedule, summarize_results  # noqa: E402
from stage_opponent_pool import (  # noqa: E402
    TEAM_ALLOWED_FILES,
    _archive_path,
    copy_public_candidate,
    copy_external_submission,
    copy_team_submission,
    prepare_arena_runtime,
    rebase_packages,
    static_scan_agent,
    validate_safe_npz,
)


def write_deck(path: Path) -> None:
    path.write_text("".join(f"{value}\n" for value in range(1, 61)), encoding="utf-8")


class StageOpponentPoolTest(unittest.TestCase):
    def test_archive_path_rejects_traversal_and_absolute_paths(self) -> None:
        for value in ("../main.py", "a/../../main.py", "/main.py"):
            with self.subTest(value=value), self.assertRaises(Experiment7Error):
                _archive_path(value)

    def test_public_copy_uses_explicit_files_and_omits_cg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "agents.zip"
            main = b"def agent(obs):\n    return []\n"
            deck = "".join(f"{value}\n" for value in range(1, 61)).encode()
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("pool/a/main.py", main)
                handle.writestr("pool/a/deck.csv", deck)
                handle.writestr("pool/a/cg/libcg.so", b"untrusted")
            candidate = {
                "prefix": "pool/a",
                "files": {
                    "main.py": __import__("hashlib").sha256(main).hexdigest(),
                    "deck.csv": __import__("hashlib").sha256(deck).hexdigest(),
                },
            }
            target = root / "staged"
            copy_public_candidate(archive, candidate, target)
            self.assertTrue((target / "main.py").is_file())
            self.assertFalse((target / "cg").exists())

    def test_static_scan_parses_but_does_not_execute_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").write_text(
                "raise RuntimeError('must never execute during staging')\n"
                "def agent(obs):\n    return []\n",
                encoding="utf-8",
            )
            write_deck(root / "deck.csv")
            scan = static_scan_agent(root)
            self.assertFalse(scan["executedAgentCode"])
            self.assertEqual(
                scan["deckCanonicalSha256"], canonical_deck_sha256(read_deck(root / "deck.csv"))
            )

    def test_static_scan_rejects_pickle_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").write_text(
                "import pickle\n"
                "def agent(obs):\n    return pickle.load(open('model.pkl', 'rb'))\n",
                encoding="utf-8",
            )
            write_deck(root / "deck.csv")
            with self.assertRaisesRegex(Experiment7Error, "pickle deserialization"):
                static_scan_agent(root)

    def test_safe_npz_rejects_object_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            safe = root / "safe.npz"
            unsafe = root / "unsafe.npz"
            np.savez(safe, weight=np.zeros((2, 2), dtype=np.float32))
            np.savez(unsafe, value=np.asarray([{"bad": True}], dtype=object))
            validate_safe_npz(safe)
            with self.assertRaises(Experiment7Error):
                validate_safe_npz(unsafe)

    def test_team_copy_only_accepts_allowlisted_top_level_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "team.tar.gz"
            files: dict[str, bytes] = {
                name: b"{}" if name.endswith(".json") else b"# source\n"
                for name in TEAM_ALLOWED_FILES
            }
            files["main.py"] = b"def agent(obs):\n    return []\n"
            files["deck.csv"] = "".join("1\n" for _ in range(60)).encode()
            model = io.BytesIO()
            np.savez(model, weight=np.zeros((1,), dtype=np.float32))
            files["deck_identity_bc.npz"] = model.getvalue()
            with tarfile.open(archive, "w:gz") as handle:
                for name, content in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    handle.addfile(info, io.BytesIO(content))
                native = b"untrusted"
                info = tarfile.TarInfo("cg/libcg.so")
                info.size = len(native)
                handle.addfile(info, io.BytesIO(native))
            # Use the helper's normal copy path but expect the production hash
            # checks to reject this synthetic fixture after proving cg omission.
            target = root / "staged"
            with self.assertRaises(Experiment7Error):
                copy_team_submission(archive, target, sha256_file(archive))
            self.assertFalse((target / "cg").exists())

    def test_external_copy_materializes_safe_top_level_agent_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "external.tar.gz"
            model = io.BytesIO()
            np.savez(model, weight=np.zeros((2,), dtype=np.float32))
            files = {
                "main.py": b"raise RuntimeError('must not run during staging')\n",
                "deck.csv": "".join(f"{value}\n" for value in range(1, 61)).encode(),
                "helper.py": b"VALUE = 1\n",
                "model.npz": model.getvalue(),
            }
            with tarfile.open(archive, "w:gz") as handle:
                for name, content in files.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(content)
                    handle.addfile(info, io.BytesIO(content))
            target = root / "staged"
            receipt = copy_external_submission(archive, target)
            self.assertEqual(receipt["fileCount"], len(files))
            self.assertTrue((target / "model.npz").is_file())
            scan = static_scan_agent(target)
            self.assertFalse(scan["executedAgentCode"])

    def test_external_copy_rejects_nested_links_and_native_assets(self) -> None:
        fixtures = {
            "nested": ("nested/main.py", tarfile.REGTYPE),
            "link": ("main.py", tarfile.SYMTYPE),
            "native": ("model.so", tarfile.REGTYPE),
        }
        for label, (name, member_type) in fixtures.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / "external.tar.gz"
                with tarfile.open(archive, "w:gz") as handle:
                    info = tarfile.TarInfo(name)
                    info.type = member_type
                    if member_type == tarfile.SYMTYPE:
                        info.linkname = "deck.csv"
                        handle.addfile(info)
                    else:
                        content = b"unsafe"
                        info.size = len(content)
                        handle.addfile(info, io.BytesIO(content))
                with self.assertRaises(Experiment7Error):
                    copy_external_submission(archive, root / "staged")

    def test_rebase_packages_verifies_relocated_agent_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agent = root / "agents" / "safe"
            agent.mkdir(parents=True)
            (agent / "main.py").write_text("def agent(obs):\n    return []\n", encoding="utf-8")
            write_deck(agent / "deck.csv")
            write_json(
                root / "staging_manifest.json",
                {
                    "agents": [
                        {
                            "name": "safe",
                            "archetype": "test",
                            "deckCanonicalSha256": canonical_deck_sha256(read_deck(agent / "deck.csv")),
                            "directorySha256": directory_sha256(agent),
                        }
                    ]
                },
            )
            output = root / "linux_packages.json"
            receipt = rebase_packages(root, output)
            self.assertEqual(receipt["agents"], 1)
            self.assertTrue(output.is_file())
            self.assertEqual(
                json.loads(output.read_text())["packages"][0]["directorySha256"],
                directory_sha256(agent),
            )

    def test_prepare_arena_runtime_rejects_source_hash_mismatch_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            learner = root / "frozen-learner"
            opponent = root / "frozen-opponent"
            for agent in (learner, opponent):
                agent.mkdir()
                (agent / "main.py").write_text("def agent(obs):\n    return []\n", encoding="utf-8")
                write_deck(agent / "deck.csv")
            packages = root / "packages.json"
            opponents = root / "opponents.json"
            write_json(
                packages,
                {
                    "packages": [
                        {
                            "name": "learner",
                            "agentDir": str(learner),
                            "directorySha256": "0" * 64,
                        }
                    ]
                },
            )
            write_json(
                opponents,
                {
                    "agents": [
                        {
                            "name": "opponent",
                            "agent_dir": str(opponent),
                            "status": "accepted",
                            "directorySha256": directory_sha256(opponent),
                        }
                    ]
                },
            )
            arena_stage = root / "arena-stage"
            with self.assertRaisesRegex(Experiment7Error, "source hash mismatch"):
                prepare_arena_runtime(packages, opponents, arena_stage, 2)
            self.assertFalse(arena_stage.exists())

    def test_prepare_arena_runtime_outputs_independent_writable_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            learner = root / "frozen-learner"
            opponent = root / "frozen-opponent"
            for agent in (learner, opponent):
                agent.mkdir()
                (agent / "main.py").write_text("def agent(obs):\n    return []\n", encoding="utf-8")
                write_deck(agent / "deck.csv")
            learner_hash = directory_sha256(learner)
            opponent_hash = directory_sha256(opponent)
            packages = root / "packages.json"
            opponents = root / "opponents.json"
            write_json(
                packages,
                {
                    "packages": [
                        {
                            "name": "learner",
                            "agentDir": str(learner),
                            "directorySha256": learner_hash,
                        }
                    ]
                },
            )
            write_json(
                opponents,
                {
                    "agents": [
                        {
                            "name": "opponent",
                            "agent_dir": str(opponent),
                            "status": "accepted",
                            "directory_sha256": opponent_hash,
                        }
                    ]
                },
            )
            arena_stage = root / "arena-stage"
            receipt = prepare_arena_runtime(packages, opponents, arena_stage, 2)
            self.assertEqual(receipt["shardCount"], 2)
            for shard_index, shard in enumerate(receipt["shards"]):
                expected_root = (arena_stage / f"runtime-shard-{shard_index}").resolve()
                self.assertEqual(Path(shard["runtimeRoot"]), expected_root)
                learners_payload = json.loads(Path(shard["learners"]["path"]).read_text())
                opponents_payload = json.loads(Path(shard["opponents"]["path"]).read_text())
                learner_row = learners_payload["agents"][0]
                opponent_row = opponents_payload["agents"][0]
                self.assertEqual(
                    Path(learner_row["agent_dir"]), expected_root / "learners" / "learner"
                )
                self.assertEqual(
                    Path(opponent_row["agent_dir"]), expected_root / "opponents" / "opponent"
                )
                self.assertEqual(learner_row["source_directory_sha256"], learner_hash)
                self.assertEqual(opponent_row["source_directory_sha256"], opponent_hash)
                self.assertTrue((Path(learner_row["agent_dir"]) / "main.py").stat().st_mode & 0o200)
                py_compile.compile(
                    str(Path(learner_row["agent_dir"]) / "main.py"), doraise=True
                )
                self.assertTrue((Path(learner_row["agent_dir"]) / "__pycache__").is_dir())
                self.assertFalse((learner / "__pycache__").exists())
            self.assertNotEqual(
                receipt["shards"][0]["runtimeRoot"], receipt["shards"][1]["runtimeRoot"]
            )
            with self.assertRaisesRegex(Experiment7Error, "refusing overwrite"):
                prepare_arena_runtime(packages, opponents, arena_stage, 2)

    def test_prepare_arena_runtime_rejects_name_escape_and_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            learner = root / "frozen-learner"
            opponent = root / "frozen-opponent"
            for agent in (learner, opponent):
                agent.mkdir()
                (agent / "main.py").write_text("def agent(obs):\n    return []\n", encoding="utf-8")
                write_deck(agent / "deck.csv")
            packages = root / "packages.json"
            opponents = root / "opponents.json"
            write_json(
                packages,
                {
                    "packages": [
                        {
                            "name": "../escape",
                            "agentDir": str(learner),
                            "directorySha256": directory_sha256(learner),
                        }
                    ]
                },
            )
            write_json(
                opponents,
                {
                    "agents": [
                        {
                            "name": "opponent",
                            "agent_dir": str(opponent),
                            "directorySha256": directory_sha256(opponent),
                        }
                    ]
                },
            )
            with self.assertRaisesRegex(Experiment7Error, "unsafe arena agent name"):
                prepare_arena_runtime(packages, opponents, root / "arena-stage", 1)

            write_json(
                packages,
                {
                    "packages": [
                        {
                            "name": "learner",
                            "agentDir": str(learner),
                            "directorySha256": directory_sha256(learner),
                        }
                    ]
                },
            )
            with mock.patch("stage_opponent_pool._is_link", side_effect=lambda path: path == learner):
                with self.assertRaisesRegex(Experiment7Error, "link rejected"):
                    prepare_arena_runtime(packages, opponents, root / "arena-stage", 1)

    def test_arena_opponents_manifest_carries_frozen_directory_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            learner = root / "learner"
            target = root / "target"
            for agent in (learner, target):
                agent.mkdir()
                (agent / "main.py").write_text("def agent(obs):\n    return []\n", encoding="utf-8")
                write_deck(agent / "deck.csv")
            packages = root / "packages.json"
            write_json(
                packages,
                {
                    "packages": [
                        {
                            "name": "learner",
                            "agentDir": str(learner),
                            "directorySha256": directory_sha256(learner),
                        }
                    ]
                },
            )
            output = root / "schedule"
            make_schedule(packages, target, output, 2, 10, "smoke", None)
            payload = json.loads((output / "opponents.json").read_text())
            self.assertEqual(
                payload["agents"][0]["directory_sha256"], directory_sha256(target)
            )

    def test_arena_external_runtime_gate_accepts_agents_without_bc_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.csv"
            rows = [
                {
                    "game_id": f"game-{seat}",
                    "learner": "external_rule_agent",
                    "opponent": "frozen_target",
                    "seed": 10 + seat,
                    "learner_seat": seat,
                    "result": "win",
                    "diagnostics_json": "[]",
                }
                for seat in (0, 1)
            ]
            write_csv(results, rows)

            external = summarize_results(
                [results], root / "external.json", "smoke", 0.0, 0.0, 0.0, 1, "external"
            )
            challenger = external["challengers"][0]
            self.assertTrue(challenger["passesRuntimeGate"])
            self.assertEqual(challenger["runtimeGateMode"], "external")
            self.assertEqual(challenger["diagnosticsRows"], 0)
            self.assertEqual(external["thresholds"]["runtimeGateMode"], "external")
            self.assertFalse(external["thresholds"]["modelCallsPositive"])
            self.assertFalse(external["thresholds"]["fallbackCallsZero"])

            default_bc = summarize_results(
                [results], root / "bc.json", "smoke", 0.0, 0.0, 0.0, 1
            )
            self.assertFalse(default_bc["challengers"][0]["passesRuntimeGate"])
            self.assertEqual(default_bc["challengers"][0]["runtimeGateMode"], "bc")

    def test_arena_external_runtime_gate_still_rejects_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results.csv"
            write_csv(
                results,
                [
                    {
                        "game_id": "failed-game",
                        "learner": "external_rule_agent",
                        "opponent": "frozen_target",
                        "seed": 10,
                        "learner_seat": 0,
                        "result": "crash",
                        "diagnostics_json": "[]",
                    }
                ],
            )
            external = summarize_results(
                [results], root / "external.json", "smoke", 0.0, 0.0, 0.0, 1, "external"
            )
            challenger = external["challengers"][0]
            self.assertEqual(challenger["failures"], 1)
            self.assertFalse(challenger["passesRuntimeGate"])


if __name__ == "__main__":
    unittest.main()
