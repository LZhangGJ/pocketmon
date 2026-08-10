from __future__ import annotations

import io
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    Experiment7Error,
    canonical_deck_sha256,
    directory_sha256,
    read_deck,
    sha256_file,
    write_json,
)
from stage_opponent_pool import (  # noqa: E402
    TEAM_ALLOWED_FILES,
    _archive_path,
    copy_public_candidate,
    copy_team_submission,
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


if __name__ == "__main__":
    unittest.main()
