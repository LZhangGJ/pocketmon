from __future__ import annotations

import argparse
import json
from pathlib import Path

from async_ppo_control import atomic_write_json, read_json, state_lock, utc_now


A02_CHAINS = ("a02_grim_large_g9", "a02_grim_large_g9_pokegear")
A08_CHAINS = ("a08_maxbelt_large_g9",)
LUCARIO_CHAINS = ("lucario_gold_exact",)
UNIVERSAL_PPO_CHAINS = ("universal_ppo_standard_1m", "universal_ppo_large_256x6")

A02_KEY_AGENT_WEIGHTS = {
    "public_alakazam_search_v9": 1.75,
    "notebook_crustle_wall": 1.75,
    "team_grim_model_a": 1.60,
    "team_grim_model_b": 1.60,
    "hard_exploiter_g0010__02_a03_606a775392ff": 1.75,
    "hard_exploiter_g0010__03_a02_cafa7652a634": 1.60,
    "hard_exploiter_g0010__05_a06_89e6155f2531": 1.75,
    "hard_exploiter_g0010__06_a05_3158359368bd": 1.75,
    "diversity_g0020__02_a03_606a775392ff": 1.75,
    "diversity_g0020__03_a02_cafa7652a634": 1.60,
    "diversity_g0020__05_a06_89e6155f2531": 1.75,
    "diversity_g0020__06_a05_3158359368bd": 1.75,
}

A02_KEY_ARCHETYPE_WEIGHTS = {
    "A02": 1.40,
    "A03": 1.50,
    "A05": 1.50,
    "A06": 1.50,
    "CRUSTLE_WALL": 1.50,
    "GREAT_TUSK_CRUSTLE_LIBRARY_OUT": 1.50,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply replay-diagnosis tactical controls to Experiment 7"
    )
    parser.add_argument("--league-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.league_root.resolve()
    league_path = root / "state/league.json"
    applied_at = utc_now()
    updated: dict[str, dict] = {}

    with state_lock(league_path.with_suffix(league_path.suffix + ".lock")):
        league = read_json(league_path)
        missing = [
            name
            for name in (*A02_CHAINS, *A08_CHAINS, *LUCARIO_CHAINS)
            if name not in league["chains"]
        ]
        if missing:
            raise KeyError(f"missing activity chains: {missing}")

        for name in A02_CHAINS:
            control = league["chains"][name].setdefault("trainingControl", {})
            rollout = control.setdefault("rollout", {})
            rollout.update(
                {
                    "tacticalShapingProfile": "a02",
                    "tacticalShapingRevision": 11,
                    "bossReservationPenalty": 0.09,
                    "bossReservationPreference": True,
                    "bossPostPlayPenalty": 0.08,
                    "bossPostPlayPreference": True,
                    "a02PoffinDeclinePenalty": 0.08,
                    "a02PoffinPreference": True,
                    "a02MunkidoriOverfillPenalty": 0.10,
                    "a02BenchBudgetPreference": True,
                    "a02OutcomeGatedOrdering": True,
                    "a02ProjectedBenchBudget": True,
                    "successorAttachPreference": True,
                    "endWithAttackPenalty": 0.06,
                    "endWithAttackPreference": True,
                    "learnerSeat1Fraction": 0.58,
                }
            )
            rollout.setdefault("agentWeights", {}).update(A02_KEY_AGENT_WEIGHTS)
            rollout.setdefault("archetypeWeights", {}).update(A02_KEY_ARCHETYPE_WEIGHTS)
            control["updatedAt"] = applied_at
            control["replayDiagnosisTuning"] = {
                "appliedAt": applied_at,
                "source": "latest_two_submissions_replay_diagnosis_20260814",
                "policy": "soft tactical preference; no hard action mask",
            }
            control["promotionTacticalGate"] = {
                "enabled": True,
                "appliesToPublishedAtOrAfter": applied_at,
                "minimumRevision": 11,
                "minimumEpisodes": 40,
                "minimumTrackedOpportunities": 10,
                "maximumAggregateErrorRate": 0.35,
                "keyMatchupGate": {
                    "enabled": True,
                    "gamesPerOpponent": 40,
                    "seatBalanced": True,
                    "maximumAggregateRegressionPp": 2.0,
                    "maximumSingleOpponentRegressionPp": 10.0,
                    "opponentPatterns": [
                        "alakazam",
                        "ogerpon",
                        "crustle",
                        "grim",
                        "dragapult",
                        "festival",
                    ],
                },
                "policy": "fail closed for revision-11 candidates when exact-snapshot evidence is missing",
            }
            updated[name] = dict(rollout)

        for name in A08_CHAINS:
            control = league["chains"][name].setdefault("trainingControl", {})
            rollout = control.setdefault("rollout", {})
            rollout.update(
                {
                    "tacticalShapingProfile": "a08",
                    "tacticalShapingRevision": 11,
                    "a08TerminalBeforeEvolveMode": "gated",
                    "a08GatedAttackPenalty": 0.10,
                    "a08SecondAttackerReward": 0.04,
                    "a08RecoveryEndPenalty": 0.10,
                    "a08RecoveryPreference": True,
                    "endWithAttackPenalty": 0.0,
                    "endWithAttackPreference": True,
                }
            )
            if name in {"a08_maxbelt", "a08_maxbelt_large_g9"}:
                rollout.update(
                    {
                        "a08MaximumBeltSupportPenalty": 0.06,
                        "a08MaximumBeltPreference": True,
                    }
                )
            else:
                rollout.update(
                    {
                        "a08MaximumBeltSupportPenalty": 0.0,
                        "a08MaximumBeltPreference": False,
                    }
                )
            control["updatedAt"] = applied_at
            control["replayDiagnosisTuning"] = {
                "appliedAt": applied_at,
                "source": "latest_two_submissions_replay_diagnosis_20260814",
                "policy": "KO/prize-gated evolve ordering and soft Belt target preference",
            }
            control["promotionTacticalGate"] = {
                "enabled": True,
                "appliesToPublishedAtOrAfter": applied_at,
                "minimumRevision": 11,
                "minimumEpisodes": 40,
                "minimumTrackedOpportunities": 10,
                "maximumAggregateErrorRate": 0.35,
                "policy": "fail closed for revision-11 candidates when exact-snapshot evidence is missing",
            }
            updated[name] = dict(rollout)

        for name in LUCARIO_CHAINS:
            control = league["chains"][name].setdefault("trainingControl", {})
            rollout = control.setdefault("rollout", {})
            rollout.update(
                {
                    "tacticalShapingProfile": "lucario",
                    "tacticalShapingRevision": 11,
                    "lucarioEvolvePenalty": 0.12,
                    "lucarioAttachPenalty": 0.06,
                    "lucarioOrderingPreference": True,
                    "successorAttachPreference": True,
                    "endWithAttackPenalty": 0.06,
                    "endWithAttackPreference": True,
                }
            )
            control["updatedAt"] = applied_at
            control["replayDiagnosisTuning"] = {
                "appliedAt": applied_at,
                "source": "latest_two_submissions_replay_diagnosis_20260814",
                "policy": (
                    "KO/prize/terminal-gated evolve and attach ordering; "
                    "soft successor preference; no hard action mask"
                ),
            }
            control["promotionTacticalGate"] = {
                "enabled": True,
                "appliesToPublishedAtOrAfter": applied_at,
                "minimumRevision": 11,
                "minimumEpisodes": 40,
                "minimumTrackedOpportunities": 10,
                "maximumAggregateErrorRate": 0.35,
                "policy": "fail closed for revision-11 candidates when exact-snapshot evidence is missing",
            }
            updated[name] = dict(rollout)

        for name in UNIVERSAL_PPO_CHAINS:
            if name not in league["chains"]:
                continue
            control = league["chains"][name].setdefault("trainingControl", {})
            rollout = control.setdefault("rollout", {})
            rollout["enabled"] = False
            control["paused"] = {
                "at": applied_at,
                "reason": "incremental BC training from frozen Universal PPO checkpoint",
                "mode": "opponent-only; no learner and no training rollout",
            }
            control["updatedAt"] = applied_at
            updated[name] = dict(rollout)

        league["trainingControlUpdatedAt"] = applied_at
        atomic_write_json(league_path, league)

    receipt = root / "monitoring/replay-diagnosis-training/20260814-controls.json"
    payload = {
        "schemaVersion": 1,
        "status": "applied",
        "appliedAt": applied_at,
        "chains": updated,
        "takesEffect": "next revision-11 rollout shard and its subsequent learner batch",
        "activeA02Chains": list(A02_CHAINS),
        "universalPpoTraining": (
            "paused; learner and training rollout disabled; opponent packages retained"
        ),
    }
    atomic_write_json(receipt, payload)
    print(json.dumps({**payload, "receipt": str(receipt)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
