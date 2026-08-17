from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_universal_bc import (  # noqa: E402
    _distributed_limits,
    combine_training_metrics,
    combine_validation_metrics,
)


class UniversalBCMultiShardTest(unittest.TestCase):
    def test_distributed_limits(self) -> None:
        self.assertEqual(_distributed_limits(10, 3), [4, 3, 3])
        self.assertEqual(_distributed_limits(0, 2), [None, None])

    def test_combines_training_metrics_by_decision_count(self) -> None:
        combined = combine_training_metrics(
            [
                (
                    "a",
                    {
                        "policyNll": 1.0,
                        "decisions": 2,
                        "seconds": 1.0,
                        "decisionsPerSecond": 2.0,
                    },
                ),
                (
                    "b",
                    {
                        "policyNll": 3.0,
                        "decisions": 6,
                        "seconds": 2.0,
                        "decisionsPerSecond": 3.0,
                    },
                ),
            ]
        )
        self.assertEqual(combined["decisions"], 8)
        self.assertEqual(combined["seconds"], 3.0)
        self.assertAlmostEqual(combined["policyNll"], 2.5)
        self.assertAlmostEqual(combined["decisionsPerSecond"], 8 / 3)

    def test_combines_validation_metrics_with_correct_denominators(self) -> None:
        combined = combine_validation_metrics(
            [
                (
                    "a",
                    {
                        "decisions": 10,
                        "policyDecisions": 4,
                        "exactIndex": 0.5,
                        "exactSemantic": 0.75,
                        "countAccuracy": 1.0,
                        "illegalPredictionCount": 0,
                        "valueBrier": 0.2,
                        "uncertainty": {
                            "meanFirstStepConfidence": 0.6,
                            "confidence60Coverage": 0.5,
                        },
                    },
                ),
                (
                    "b",
                    {
                        "decisions": 30,
                        "policyDecisions": 6,
                        "exactIndex": 1.0,
                        "exactSemantic": 0.5,
                        "countAccuracy": 0.5,
                        "illegalPredictionCount": 1,
                        "valueBrier": 0.4,
                        "uncertainty": {
                            "meanFirstStepConfidence": 0.8,
                            "confidence60Coverage": 0.9,
                        },
                    },
                ),
            ]
        )
        self.assertEqual(combined["decisions"], 40)
        self.assertEqual(combined["policyDecisions"], 10)
        self.assertAlmostEqual(combined["exactIndex"], 0.8)
        self.assertAlmostEqual(combined["exactSemantic"], 0.6)
        self.assertAlmostEqual(combined["valueBrier"], 0.35)
        self.assertAlmostEqual(
            combined["uncertainty"]["meanFirstStepConfidence"], 0.75
        )
        self.assertEqual(combined["illegalPredictionCount"], 1)


if __name__ == "__main__":
    unittest.main()
