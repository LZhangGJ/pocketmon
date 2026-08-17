import unittest

from scripts.summarize_league_round import summarize


class SummarizeLeagueRoundTests(unittest.TestCase):
    def test_ranks_and_verifies_paired_seats(self):
        rows = [
            {"learner": "a", "opponent": "x", "seed": "1", "learner_seat": "0", "result": "win", "engine_seed_controlled": "false"},
            {"learner": "a", "opponent": "x", "seed": "1", "learner_seat": "1", "result": "win", "engine_seed_controlled": "false"},
            {"learner": "b", "opponent": "x", "seed": "2", "learner_seat": "0", "result": "loss", "engine_seed_controlled": "false"},
            {"learner": "b", "opponent": "x", "seed": "2", "learner_seat": "1", "result": "draw", "engine_seed_controlled": "false"},
        ]
        report = summarize(rows)
        self.assertEqual(report["learners"][0]["learner"], "a")
        self.assertFalse(report["incomplete_pairs"])
        self.assertFalse(report["engine_seed_controlled"])


if __name__ == "__main__":
    unittest.main()
