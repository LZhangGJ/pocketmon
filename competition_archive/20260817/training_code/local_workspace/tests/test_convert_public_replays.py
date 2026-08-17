from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.convert_public_replays import main


def observation() -> dict:
    return {
        "current": {"result": -1, "yourIndex": 0, "players": [{}, {}]},
        "select": {
            "option": [{"type": 3, "cardId": 100}],
            "minCount": 1,
            "maxCount": 1,
        },
    }


def mismatched_reward_replay() -> dict:
    return {
        "info": {"EpisodeId": 99},
        "rewards": [-1, 1],
        "steps": [
            [
                {"action": None, "status": "ACTIVE", "observation": observation(), "reward": 0},
                {"action": None, "status": "ACTIVE", "observation": observation(), "reward": 0},
            ],
            [
                {"action": [0], "status": "DONE", "observation": observation(), "reward": 1},
                {"action": [0], "status": "DONE", "observation": observation(), "reward": -1},
            ],
        ],
    }


class ConvertPublicReplaysTests(unittest.TestCase):
    def test_reward_mismatch_blocks_output_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            replay_directory = root / "replays" / "2026-07-20"
            replay_directory.mkdir(parents=True)
            (replay_directory / "99.json").write_text(
                json.dumps(mismatched_reward_replay()),
                encoding="utf-8",
            )
            output = root / "processed.jsonl.gz"
            report = root / "report.json"
            arguments = [
                "convert_public_replays.py",
                "--input-root",
                str(root / "replays"),
                "--date",
                "2026-07-20",
                "--output",
                str(output),
                "--report",
                str(report),
            ]

            with patch.object(sys, "argv", arguments):
                with self.assertRaisesRegex(RuntimeError, "conversion gate failed"):
                    main()

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertFalse(payload["gate_passed"])
            self.assertEqual(payload["reward_mismatches"], 1)
            self.assertEqual(payload["episodes_missing_winner"], 0)
            self.assertEqual(payload["load_errors"], 0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
