from __future__ import annotations

import unittest

from universal_deck_cohort import select_cohort


class UniversalDeckCohortTests(unittest.TestCase):
    def test_tier_probabilities_and_selection_are_deterministic(self) -> None:
        rows = []
        tier_weights = {"A": 0.74, "B": 0.13, "C": 0.129, "D": 0.001}
        for tier, count in {"A": 45, "B": 27, "C": 57, "D": 4}.items():
            for index in range(count):
                rows.append(
                    {
                        "name": f"{tier}{index}",
                        "deckSha256": f"{tier}{index:03d}",
                        "evidenceTier": tier,
                        "samplingWeight": tier_weights[tier] / count,
                    }
                )
        first, receipt = select_cohort(rows, size=20, seed=1234)
        second, second_receipt = select_cohort(rows, size=20, seed=1234)
        self.assertEqual(first, second)
        self.assertEqual(receipt, second_receipt)
        self.assertEqual(len(first), 20)
        self.assertEqual(len({row["deckSha256"] for row in first}), 20)
        by_tier = {
            tier: sum(row["samplingWeight"] for row in first if row["evidenceTier"] == tier)
            for tier in "ABCD"
        }
        for tier, expected in receipt["tierProbabilities"].items():
            if receipt["tierCounts"].get(tier, 0):
                self.assertAlmostEqual(by_tier[tier], expected)


if __name__ == "__main__":
    unittest.main()
