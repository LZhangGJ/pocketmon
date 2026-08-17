from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from promote_ppo_to_frozen_pool import (
    build_frozen_pool,
    evaluate_round,
    frozen_agent,
    update_candidate,
)


class PromotionGateTests(unittest.TestCase):
    def candidate(self) -> dict:
        return {
            "chain": "a02_grim_g247",
            "generation": 275,
            "snapshotId": "a02_grim_g247-g000275-deadbeef",
            "checkpointSha256": "c" * 64,
            "packageManifest": "/tmp/packages.json",
            "agentDir": "/tmp/agent",
            "directorySha256": "d" * 64,
            "deckSha256": "e" * 64,
            "archetypeId": "A02",
            "archetypeLabel": "A02",
            "status": "testing",
            "firstAdmission": False,
            "passRounds": [],
            "observations": [],
        }

    def report(self, round_id: str, *, delta: float = 0.0, direct: int = 40) -> dict:
        return {
            "status": "complete",
            "roundId": round_id,
            "_path": f"/tmp/{round_id}/report.json",
            "chains": {
                "a02_grim_g247": {
                    "snapshotId": "a02_grim_g247-g000275-deadbeef",
                    "frozenAggregate": {
                        "completed": 144,
                        "failures": 0,
                        "scoreRate": 0.55,
                    },
                    "directVsUniversalBc": {
                        "completed": direct,
                        "failures": 0,
                        "scoreRate": 0.525,
                    },
                    "deltaVsPrevious": delta,
                    "seatGap": 0.02,
                }
            },
        }

    def test_requires_two_distinct_passing_rounds(self) -> None:
        candidate = self.candidate()
        update_candidate(
            candidate,
            [self.report("r1"), self.report("r2")],
            min_frozen_games=40,
            min_direct_games=40,
            max_regression=0.02,
            initial_min_score=0.5,
            required_passes=2,
        )
        self.assertEqual(candidate["status"], "passed")
        self.assertEqual(candidate["passRounds"], ["r1", "r2"])

    def test_stale_snapshot_is_ignored(self) -> None:
        candidate = self.candidate()
        report = self.report("r1")
        report["chains"]["a02_grim_g247"]["snapshotId"] = "stale"
        self.assertIsNone(
            evaluate_round(
                candidate,
                report,
                min_frozen_games=40,
                min_direct_games=40,
                max_regression=0.02,
                initial_min_score=0.5,
            )
        )

    def test_failed_round_resets_consecutive_passes(self) -> None:
        candidate = self.candidate()
        update_candidate(
            candidate,
            [self.report("r1"), self.report("r2", direct=20)],
            min_frozen_games=40,
            min_direct_games=40,
            max_regression=0.02,
            initial_min_score=0.5,
            required_passes=2,
        )
        self.assertEqual(candidate["status"], "testing")
        self.assertEqual(candidate["passRounds"], [])
        self.assertIn("insufficient_direct_games", candidate["observations"][-1]["reasons"])

    def test_regression_over_two_points_fails(self) -> None:
        observation = evaluate_round(
            self.candidate(),
            self.report("r1", delta=-0.021),
            min_frozen_games=40,
            min_direct_games=40,
            max_regression=0.02,
            initial_min_score=0.5,
        )
        self.assertFalse(observation["passed"])
        self.assertIn("regressed_over_limit", observation["reasons"])

    def test_first_admission_cannot_evade_absolute_floor_on_later_round(self) -> None:
        candidate = self.candidate()
        candidate["firstAdmission"] = True
        report = self.report("r2", delta=0.0)
        report["chains"]["a02_grim_g247"]["frozenAggregate"]["scoreRate"] = 0.49
        observation = evaluate_round(
            candidate,
            report,
            min_frozen_games=40,
            min_direct_games=40,
            max_regression=0.02,
            initial_min_score=0.5,
        )
        self.assertFalse(observation["passed"])
        self.assertIn("initial_score_below_floor", observation["reasons"])

    def test_revision7_candidate_fails_closed_without_exact_snapshot_evidence(self) -> None:
        candidate = self.candidate()
        candidate["tacticalGate"] = {
            "appliesToCandidate": True,
            "minimumRevision": 7,
            "minimumEpisodes": 40,
            "minimumTrackedOpportunities": 10,
            "maximumAggregateErrorRate": 0.35,
        }
        observation = evaluate_round(
            candidate,
            self.report("r1"),
            min_frozen_games=40,
            min_direct_games=40,
            max_regression=0.02,
            initial_min_score=0.5,
            tactical_evidence=None,
        )
        self.assertFalse(observation["passed"])
        self.assertIn(
            "missing_exact_snapshot_tactical_evidence", observation["reasons"]
        )

    def test_revision7_candidate_accepts_sufficient_tactical_evidence(self) -> None:
        candidate = self.candidate()
        candidate["tacticalGate"] = {
            "appliesToCandidate": True,
            "minimumRevision": 7,
            "minimumEpisodes": 40,
            "minimumTrackedOpportunities": 10,
            "maximumAggregateErrorRate": 0.35,
        }
        evidence = {
            "snapshots": {
                candidate["chain"]: {
                    candidate["snapshotId"]: {
                        "minimumRevision": 7,
                        "episodes": 60,
                        "trackedOpportunities": 20,
                        "aggregateErrorRate": 0.20,
                    }
                }
            }
        }
        observation = evaluate_round(
            candidate,
            self.report("r1"),
            min_frozen_games=40,
            min_direct_games=40,
            max_regression=0.02,
            initial_min_score=0.5,
            tactical_evidence=evidence,
        )
        self.assertTrue(observation["passed"])

    def test_pool_keeps_one_promoted_version_per_chain(self) -> None:
        candidate = self.candidate()
        candidate["passRounds"] = ["r1", "r2"]
        agent = frozen_agent(candidate)
        source = {
            "agents": [
                {"name": "base"},
                {
                    "name": "old",
                    "pool_status": "frozen_ppo_version",
                    "ppo_chain": "a02_grim_g247",
                },
            ]
        }
        pool = build_frozen_pool(
            source,
            {
                "a02_grim_g247": {"agent": agent},
                "_sourcePool": "/tmp/base.json",
            },
        )
        self.assertEqual([row["name"] for row in pool["agents"]], ["base", agent["name"]])


if __name__ == "__main__":
    unittest.main()
