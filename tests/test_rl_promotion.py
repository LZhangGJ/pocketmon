from __future__ import annotations

import unittest

from rl.promotion import build_promotion_schedule, evaluate_promotion


def result_rows(candidate_wins: int, parent_wins: int, head_wins: int) -> list[dict]:
    rows = []
    for learner, wins in (("candidate", candidate_wins), ("parent", parent_wins)):
        for index in range(4):
            rows.append({
                "learner": learner,
                "opponent": "public",
                "seed": index // 2,
                "learner_seat": index % 2,
                "result": "win" if index < wins else "loss",
            })
    for index in range(20):
        rows.append({
            "learner": "candidate",
            "opponent": "parent",
            "seed": 100 + index // 2,
            "learner_seat": index % 2,
            "result": "win" if index < head_wins else "loss",
        })
    return rows


class PromotionTests(unittest.TestCase):
    def test_schedule_matches_public_seeds_and_adds_head_to_head(self) -> None:
        rows = build_promotion_schedule(
            candidate="candidate", parent="parent", public_opponents=["a", "b"],
            games_per_public=4, parent_games=20, seed=7,
        )
        self.assertEqual(len(rows), 36)
        candidate_public = {(r["opponent"], r["seed"], r["learner_seat"]) for r in rows if r["learner"] == "candidate" and r["opponent"] in {"a", "b"}}
        parent_public = {(r["opponent"], r["seed"], r["learner_seat"]) for r in rows if r["learner"] == "parent"}
        self.assertEqual(candidate_public, parent_public)

    def test_promotes_only_when_parent_and_public_gates_pass(self) -> None:
        promoted = evaluate_promotion(
            result_rows(candidate_wins=3, parent_wins=2, head_wins=14),
            candidate="candidate", parent="parent", public_opponents=["public"],
            min_head_to_head_wilson=0.30, max_seat_gap=1.0,
        )
        self.assertTrue(promoted["promote"])
        rejected = evaluate_promotion(
            result_rows(candidate_wins=1, parent_wins=3, head_wins=14),
            candidate="candidate", parent="parent", public_opponents=["public"],
            min_head_to_head_wilson=0.30, max_seat_gap=1.0,
        )
        self.assertFalse(rejected["promote"])
        self.assertFalse(rejected["checks"]["public_delta"])


if __name__ == "__main__":
    unittest.main()
