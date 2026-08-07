import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts.download_kaggle_intelligence import (
    message_fingerprint,
    normalize_notebook_candidates,
    select_live_notebook_candidates,
)


class DownloadKaggleIntelligenceTests(unittest.TestCase):
    def test_candidate_order_and_metadata_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "frontier.json"
            path.write_text(json.dumps({"notebooks": [
                {"ref": "new/agent", "public_score_observed": 800},
                {"ref": "old/agent", "public_score_observed": 950},
            ]}), encoding="utf-8")
            rows = normalize_notebook_candidates(path)
            self.assertEqual([row["ref"] for row in rows], ["new/agent", "old/agent"])
            self.assertEqual(rows[0]["public_score_observed"], 800)

    def test_duplicate_refs_are_not_redownloaded(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "frontier.json"
            path.write_text(json.dumps([
                {"ref": "a/b", "updated_label_observed": "1h"},
                {"ref": "a/b", "updated_label_observed": "2h"},
            ]), encoding="utf-8")
            self.assertEqual(len(normalize_notebook_candidates(path)), 1)

    def test_message_content_change_moves_cursor(self):
        one = message_fingerprint({"id": 1, "content": "first"})
        two = message_fingerprint({"id": 1, "content": "edited"})
        self.assertNotEqual(one, two)

    def test_live_selection_prefers_fresh_score_and_adds_diverse_recent_method(self):
        score = [
            {"ref": "old/global-best", "title": "old", "lastRunTime": "2026-06-01T00:00:00"},
            {"ref": "fresh/strong", "title": "Strong agent", "lastRunTime": "2026-08-06T00:00:00"},
        ]
        recent = [
            {"ref": "fresh/eda", "title": "EDA", "lastRunTime": "2026-08-07T00:00:00"},
            {"ref": "fresh/muzero", "title": "MuZero RL agent", "lastRunTime": "2026-08-07T00:00:00"},
        ]
        rows = select_live_notebook_candidates(
            score,
            recent,
            now=datetime(2026, 8, 7, tzinfo=timezone.utc),
            freshness_days=21,
            limit=2,
        )
        self.assertEqual([row["ref"] for row in rows], ["fresh/strong", "fresh/muzero"])
        self.assertNotIn("old/global-best", [row["ref"] for row in rows])


if __name__ == "__main__":
    unittest.main()
