import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_league_schedule import classify_result, load_agents, row_key


class RunLeagueScheduleTests(unittest.TestCase):
    def test_result_is_from_learner_perspective(self):
        self.assertEqual(classify_result({"result": 0}, 0), "win")
        self.assertEqual(classify_result({"result": 0}, 1), "loss")
        self.assertEqual(classify_result({"result": 2}, 1), "draw")

    def test_row_key_normalizes_csv_values(self):
        self.assertEqual(
            row_key({"learner": "a", "opponent": "b", "seed": "7", "learner_seat": "1"}),
            ("a", "b", 7, 1),
        )

    def test_manifest_accepts_list_and_snapshot_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "agents.json"
            path.write_text(json.dumps([{"name": "a", "agent_dir": "."}]), encoding="utf-8")
            self.assertIn("a", load_agents(path))
            path.write_text(
                json.dumps({"agents": [{"name": "b", "path": ".", "status": "accepted"}]}),
                encoding="utf-8",
            )
            self.assertIn("b", load_agents(path))


if __name__ == "__main__":
    unittest.main()
