from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.collect_ppo_rollouts import choose_frozen_opponent, read_deck


class FrozenLeagueRolloutTests(unittest.TestCase):
    def test_deck_read_retries_during_atomic_refresh_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deck.csv"
            path.write_text("1\n" * 60, encoding="utf-8")
            with patch.object(Path, "read_text", side_effect=["1\n", "2\n" * 60]), patch(
                "scripts.collect_ppo_rollouts.time.sleep"
            ) as sleep:
                deck = read_deck(path, attempts=2, retry_seconds=0.01)
        self.assertEqual(deck, [2] * 60)
        sleep.assert_called_once_with(0.01)

    def setUp(self) -> None:
        self.pool = [
            {"name": "public", "league_role": "public"},
            {"name": "parent", "league_role": "population"},
        ]

    def test_population_fraction_one_always_uses_frozen_parent(self) -> None:
        chosen = choose_frozen_opponent(self.pool, random.Random(7), 1.0)
        self.assertEqual(chosen["name"], "parent")

    def test_population_fraction_zero_uses_public_opponent(self) -> None:
        chosen = choose_frozen_opponent(self.pool, random.Random(7), 0.0)
        self.assertEqual(chosen["name"], "public")

    def test_missing_preferred_role_falls_back_to_available_frozen_opponent(self) -> None:
        chosen = choose_frozen_opponent([self.pool[0]], random.Random(7), 1.0)
        self.assertEqual(chosen["name"], "public")

    def test_invalid_fraction_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "fraction"):
            choose_frozen_opponent(self.pool, random.Random(7), 1.01)


if __name__ == "__main__":
    unittest.main()
