from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_baseline_dataset import build_rows


class BaselineDatasetTests(unittest.TestCase):
    def test_labels_use_previous_observation_and_only_winner_policy_rows(self) -> None:
        replay = {
            "info": {"EpisodeId": 9},
            "steps": [
                [
                    {"action": None, "observation": {"current": {"result": -1}, "select": {"option": [{}, {}, {}], "minCount": 1, "maxCount": 1}}},
                    {"action": None, "observation": {"current": {"result": -1}, "select": {"option": [{}, {}], "minCount": 1, "maxCount": 1}}},
                ],
                [
                    {"action": [2], "observation": {"current": {"result": 0}, "select": {"option": [{}], "minCount": 1, "maxCount": 1}}},
                    {"action": [1], "observation": {"current": {"result": 0}, "select": {"option": [{}], "minCount": 1, "maxCount": 1}}},
                ],
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "9.json"
            path.write_text(json.dumps(replay), encoding="utf-8")
            rows = build_rows([path])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["agent_idx"], 0)
        self.assertEqual(rows[0]["option_count"], 3)
        self.assertEqual(rows[0]["target_action"], "[2]")


if __name__ == "__main__":
    unittest.main()
