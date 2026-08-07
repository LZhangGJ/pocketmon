from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.merge_replay_datasets import merge


def write_gzip(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def replay_row(episode: str, source_sha: str, date: str) -> dict:
    return {
        "schema_version": 2,
        "episode_id": episode,
        "source_sha256": source_sha,
        "action": [0],
        "policy_weight": 1.0,
        "value_weight": 1.0,
        "manifest": {"create_time": f"{date}T00:00:00Z", "avg_score": "1000"},
        "observation": {"select": {"option": [{"type": 1}], "minCount": 1, "maxCount": 1}},
    }


def decks(episode: str) -> list[dict]:
    return [
        {"episode_id": episode, "player": player, "deck": [player + 1] * 60, "source_sha256": "x" * 64}
        for player in (0, 1)
    ]


class MergeReplayDatasetsTests(unittest.TestCase):
    def test_merges_days_and_skips_identical_cross_input_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay_a, replay_b = root / "a.jsonl.gz", root / "b.jsonl.gz"
            deck_a, deck_b = root / "da.jsonl.gz", root / "db.jsonl.gz"
            write_gzip(replay_a, [replay_row("1", "a" * 64, "2026-08-05")])
            write_gzip(replay_b, [replay_row("1", "a" * 64, "2026-08-05"), replay_row("2", "b" * 64, "2026-08-06")])
            write_gzip(deck_a, decks("1"))
            write_gzip(deck_b, decks("1") + decks("2"))
            report = merge(
                [replay_a, replay_b],
                [deck_a, deck_b],
                root / "merged.jsonl.gz",
                root / "merged-decks.jsonl.gz",
                root / "report.json",
            )
            self.assertTrue(report["gate_passed"])
            self.assertEqual(report["episodes"], 2)
            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["duplicate_rows_skipped"], 1)
            self.assertEqual(report["invalid_actions"], 0)
            self.assertEqual(report["deck_entries"], 4)

    def test_rejects_conflicting_episode_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay_a, replay_b = root / "a.jsonl.gz", root / "b.jsonl.gz"
            deck_a, deck_b = root / "da.jsonl.gz", root / "db.jsonl.gz"
            write_gzip(replay_a, [replay_row("1", "a" * 64, "2026-08-05")])
            write_gzip(replay_b, [replay_row("1", "b" * 64, "2026-08-06")])
            write_gzip(deck_a, decks("1"))
            write_gzip(deck_b, decks("1"))
            with self.assertRaisesRegex(ValueError, "conflicting source SHA"):
                merge(
                    [replay_a, replay_b],
                    [deck_a, deck_b],
                    root / "merged.jsonl.gz",
                    root / "merged-decks.jsonl.gz",
                    root / "report.json",
                )


if __name__ == "__main__":
    unittest.main()
