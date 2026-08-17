import unittest

from scripts.league_v1 import build_named_schedule, build_schedule, evaluate, wilson_lower


class LeagueV1Tests(unittest.TestCase):
    def test_schedule_pairs_seats_with_common_seed(self):
        rows = build_schedule(10, ["public_a"], 4, 7)
        self.assertEqual(len(rows), 40)
        learner = [r for r in rows if r.learner == "learner_00"]
        pairs = {(r.seed, r.learner_seat) for r in learner}
        self.assertEqual(pairs, {(7, 0), (7, 1), (8, 0), (8, 1)})

    def test_wilson_is_conservative(self):
        self.assertLess(wilson_lower(60, 100), 0.60)
        self.assertGreater(wilson_lower(300, 400), 0.70)

    def test_named_confirmation_schedule(self):
        rows = build_named_schedule(["a", "b"], ["x"], 4, 9)
        self.assertEqual(len(rows), 8)
        self.assertEqual({row.learner for row in rows}, {"a", "b"})

    def test_all_ten_must_pass(self):
        config = {"evaluation": {"min_public_win_rate": .6, "min_lower_wilson_bound": .5, "max_failure_rate": 0}, "promotion": {"required_qualified_learners": 10}}
        rows = []
        for i in range(10):
            rows += [{"learner": f"learner_{i:02d}", "opponent": "p", "result": "win"}] * 80
            rows += [{"learner": f"learner_{i:02d}", "opponent": "p", "result": "loss"}] * 20
        self.assertTrue(evaluate(rows, config)["promote"])
        rows[-1]["result"] = "illegal"
        self.assertFalse(evaluate(rows, config)["promote"])


if __name__ == "__main__":
    unittest.main()
