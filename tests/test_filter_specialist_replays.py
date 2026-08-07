from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.filter_specialist_replays import deck_similarity, filter_replays


class SpecialistReplayFilterTests(unittest.TestCase):
    def test_multiset_similarity_respects_card_counts(self) -> None:
        left = [1] * 30 + [2] * 30
        right = [1] * 20 + [2] * 30 + [3] * 10
        self.assertAlmostEqual(deck_similarity(left, right), 50 / 60)

    def test_filters_by_acting_players_submitted_deck(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "deck.csv"
            target.write_text("".join("1\n" for _ in range(60)), encoding="utf-8")
            deck_map = root / "decks.jsonl.gz"
            with gzip.open(deck_map, "wt", encoding="utf-8") as handle:
                for player, card_id in ((0, 1), (1, 2)):
                    handle.write(json.dumps({
                        "episode_id": "e1", "player": player, "deck": [card_id] * 60,
                    }) + "\n")
            source = root / "source.jsonl.gz"
            with gzip.open(source, "wt", encoding="utf-8") as handle:
                for player in (0, 1):
                    handle.write(json.dumps({
                        "schema_version": 2,
                        "episode_id": "e1",
                        "player": player,
                        "policy_weight": 1.0,
                        "value_weight": 1.0,
                    }) + "\n")
            output = root / "specialist.jsonl.gz"
            audit_path = root / "audit.json"
            audit = filter_replays(
                input_path=source,
                deck_map_path=deck_map,
                target_deck_path=target,
                output_path=output,
                audit_path=audit_path,
                min_similarity=1.0,
                min_episode_players=1,
            )
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual([row["player"] for row in rows], [0])
            self.assertEqual(audit["output"]["rows"], 1)
            self.assertEqual(audit["selection"]["observed_episode_players"], 1)

    def test_optional_cap_is_deterministic_and_label_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "deck.csv"
            target.write_text("".join("1\n" for _ in range(60)), encoding="utf-8")
            deck_map = root / "decks.jsonl.gz"
            source = root / "source.jsonl.gz"
            with gzip.open(deck_map, "wt", encoding="utf-8") as decks, gzip.open(
                source, "wt", encoding="utf-8"
            ) as rows:
                for episode in range(4):
                    decks.write(json.dumps({
                        "episode_id": f"e{episode}", "player": 0, "deck": [1] * 60,
                    }) + "\n")
                    rows.write(json.dumps({
                        "schema_version": 2,
                        "episode_id": f"e{episode}",
                        "player": 0,
                        "policy_weight": 1.0,
                        "value_weight": 1.0,
                    }) + "\n")
            output = root / "specialist.jsonl.gz"
            audit_path = root / "audit.json"
            audit = filter_replays(
                input_path=source,
                deck_map_path=deck_map,
                target_deck_path=target,
                output_path=output,
                audit_path=audit_path,
                min_similarity=1.0,
                min_episode_players=1,
                max_episode_players=2,
                selection_seed=11,
            )
            self.assertEqual(audit["selection"]["eligible_episode_players"], 4)
            self.assertEqual(audit["selection"]["selected_episode_players"], 2)
            self.assertFalse(audit["selection"]["selection_uses_outcome_or_action"])


if __name__ == "__main__":
    unittest.main()
