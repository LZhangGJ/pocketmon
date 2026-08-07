import unittest
from collections import Counter

from scripts.mutate_legal_decks import generate, validate_deck


class MutateLegalDeckTests(unittest.TestCase):
    def setUp(self):
        self.cards = {
            1: {"cardId": 1, "name": "Basic Energy", "cardType": 5, "basic": False},
            10: {"cardId": 10, "name": "Basic A", "cardType": 0, "basic": True},
            11: {"cardId": 11, "name": "Basic B", "cardType": 0, "basic": True},
            20: {"cardId": 20, "name": "Trainer A", "cardType": 2, "basic": False},
            21: {"cardId": 21, "name": "Trainer B", "cardType": 2, "basic": False},
        }
        self.base = [1] * 48 + [10] * 4 + [20] * 4 + [21] * 4

    def test_generates_unique_legal_sixty_card_candidates(self):
        rows = generate(self.base, self.cards, Counter({11: 5, 20: 3, 21: 3, 1: 2}), 5, 7, 2)
        self.assertEqual(len(rows), 5)
        self.assertEqual(len({row["sha256"] for row in rows}), 5)
        self.assertTrue(all(len(row["deck"]) == 60 and not validate_deck(row["deck"], self.cards) for row in rows))

    def test_rejects_more_than_four_by_name(self):
        bad = list(self.base)
        bad[0] = 20
        self.assertTrue(any("four copies" in error for error in validate_deck(bad, self.cards)))


if __name__ == "__main__":
    unittest.main()
