from __future__ import annotations

import random
import unittest

from scripts.collect_ppo_rollouts import choose_frozen_opponent


class FrozenLeagueRolloutTests(unittest.TestCase):
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
