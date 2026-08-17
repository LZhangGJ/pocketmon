import unittest
from collections import Counter
from pathlib import Path

from scripts.generate_multibase_deck_population import (
    BaseDeck,
    generate_population,
    multiset_fingerprint,
)
from scripts.mutate_legal_decks import validate_deck


class GenerateMultibaseDeckPopulationTests(unittest.TestCase):
    def setUp(self):
        self.cards = {
            1: {"cardId": 1, "name": "Basic Energy", "cardType": 5, "basic": False},
            2: {"cardId": 2, "name": "Basic Water Energy", "cardType": 5, "basic": False},
            10: {"cardId": 10, "name": "Basic A", "cardType": 0, "basic": True},
            11: {"cardId": 11, "name": "Basic B", "cardType": 0, "basic": True},
            12: {"cardId": 12, "name": "Basic C", "cardType": 0, "basic": True},
            20: {"cardId": 20, "name": "Trainer A", "cardType": 2, "basic": False},
            21: {"cardId": 21, "name": "Trainer B", "cardType": 2, "basic": False},
            22: {"cardId": 22, "name": "Trainer C", "cardType": 2, "basic": False},
            23: {"cardId": 23, "name": "Trainer D", "cardType": 2, "basic": False},
        }
        deck_a = [1] * 48 + [10] * 4 + [20] * 4 + [21] * 4
        deck_b = [2] * 48 + [11] * 4 + [22] * 4 + [23] * 4
        self.bases = [
            BaseDeck("a", "alpha", Path("a.csv"), tuple(deck_a)),
            BaseDeck("b", "beta", Path("b.csv"), tuple(deck_b)),
        ]

    def test_population_is_deterministic_unique_and_legal(self):
        frequencies = Counter({1: 4, 2: 4, 10: 2, 11: 2, 12: 2, 20: 2, 21: 2, 22: 2, 23: 2})
        first = generate_population(self.bases, self.cards, frequencies, 12, 42, 3)
        second = generate_population(self.bases, self.cards, frequencies, 12, 42, 3)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(len({multiset_fingerprint(row["deck"]) for row in first}), 12)
        self.assertTrue(all(not validate_deck(row["deck"], self.cards) for row in first))
        self.assertEqual({row["method"] for row in first[:2]}, {"base"})
        self.assertIn("crossover", {row["method"] for row in first[2:]})


if __name__ == "__main__":
    unittest.main()
