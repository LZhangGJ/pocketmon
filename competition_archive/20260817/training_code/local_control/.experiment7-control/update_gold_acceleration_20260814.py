from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")
PLAN = ROOT / "control/gold-acceleration-plan.json"
LATEST = ROOT / "monitoring/gold-acceleration/latest.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    observed = datetime.now(timezone.utc).isoformat()
    ppo = {
        "a02_grim_g247": {
            "generation": 260, "generationDelta": 6, "shards": 446, "shardDelta": 204,
            "episodes": 8920, "decisions": 915677, "decisionDelta": 423119,
            "external": "4619-2956", "updatesDelta": 6, "failedDelta": 0,
        },
        "a02_grim_g247_pokegear": {
            "generation": 260, "generationDelta": 6, "shards": 445, "shardDelta": 204,
            "episodes": 8900, "decisions": 936992, "decisionDelta": 428161,
            "external": "4787-2805", "updatesDelta": 6, "failedDelta": 0,
        },
        "a08_maxbelt": {
            "generation": 286, "generationDelta": 4, "shards": 457, "shardDelta": 202,
            "episodes": 9140, "decisions": 621820, "decisionDelta": 273496,
            "external": "4080-3214", "updatesDelta": 4, "failedDelta": 0,
        },
        "a08_rabsca": {
            "generation": 286, "generationDelta": 4, "shards": 455, "shardDelta": 203,
            "episodes": 9100, "decisions": 586773, "decisionDelta": 261792,
            "external": "3849-3405", "updatesDelta": 4, "failedDelta": 0,
        },
    }
    snapshot = {
        "schemaVersion": 1,
        "observedAt": observed,
        "engine_seed_controlled": False,
        "ppo": ppo,
        "bcScreening": {
            "currentFrozen40": "complete",
            "standardFrozen40": "complete",
            "largeFrozen40": "complete",
            "gamesPerProfile": 9160,
            "candidateFrozenAverages": {
                "current_bc": 0.5649553571428572,
                "standard_1m": 0.6030133928571429,
                "large_256x6": 0.6316964285714286,
            },
            "standardDirectRound1": {
                "status": "complete", "games": 160, "wins": 95, "losses": 65,
                "scoreRate": 0.59375, "round2": "running",
            },
            "largeDirectRound1": {
                "status": "complete", "games": 160, "wins": 84, "losses": 76,
                "scoreRate": 0.525, "round2": "running",
            },
            "admission": "not_yet_allowed_waiting_two_direct_rounds",
        },
        "fixedArena": {
            "previousRound": "stale_retired_five_lines_20260813T112045Z",
            "currentFourLineRound": "running_on_doraemon03",
        },
        "replay": {
            "latestCacheDate": "2026-08-12",
            "latestWindowEnd": "2026-08-12",
            "date0813Present": False,
            "note": "04:00 JST, before daily 11:00 publication check window",
        },
        "audits": {
            "a02": "Boss turn-reservation implementation ready; focused tests passed",
            "a08": "control/end_only/gated evolve-order implementation ready; focused tests passed",
            "combinedTests": "10/10 local and remote temp runtime",
        },
        "actions": [
            "launched standard and large same-deck candidate-vs-current BC paired-seat direct gates, two rounds",
            "launched current four-line fixed Arena on doraemon03 without duplicate",
            "validated combined tactical shaping revision 4 in local and remote runtime temp paths",
            "started watcher PID 667713 to launch current/large frozen40 round2 in parallel, then standard round2",
        ],
    }
    write(LATEST, snapshot)

    plan = load(PLAN)
    items = {item["experimentId"]: item for item in plan["items"]}
    item = items["ppo-active-four"]
    item["status"] = "running_healthy"
    item["receipt"] = {"observedAt": observed, "chains": ppo, "failedUpdateDelta": 0}
    item["nextAction"] = "continue rollout; current four-line fixed Arena is running"

    standard = items["bc-standard-replacement"]
    standard["status"] = "direct_round1_pass_round2_running"
    standard["startCommand"] = (
        "/homes/lzhang/run_bc_direct_replacement_gate_20260814.py "
        "--profile standard_1m --rounds 2 --games-per-deck 40"
    )
    standard["receipt"] = {
        "observedAt": observed,
        "parity": "passed", "smoke": "passed", "frozen40Games": 9160,
        "frozenAverage": 0.6030133928571429, "currentFrozenAverage": 0.5649553571428572,
        "directRound1": {"games": 160, "wins": 95, "losses": 65, "scoreRate": 0.59375},
    }
    standard["nextAction"] = "finish paired-seat direct round2 and frozen40 round2, then create admission decision receipt"

    large = items["bc-large-replacement"]
    large["status"] = "direct_round1_pass_round2_running"
    large["startCommand"] = (
        "/homes/lzhang/run_bc_direct_replacement_gate_20260814.py "
        "--profile large_256x6 --rounds 2 --games-per-deck 40"
    )
    large["receipt"] = {
        "observedAt": observed,
        "parity": "passed", "smoke": "passed", "frozen40Games": 9160,
        "frozenAverage": 0.6316964285714286, "currentFrozenAverage": 0.5649553571428572,
        "directRound1": {"games": 160, "wins": 84, "losses": 76, "scoreRate": 0.525},
    }
    large["nextAction"] = "finish direct round2 and frozen40 round2; large remains admission priority"

    for experiment_id, status, next_action in (
        (
            "loss-audit-a02", "analysis_and_implementation_complete",
            "launch independent g260 A/B after remote worktree integration test",
        ),
        (
            "loss-audit-a08", "analysis_and_implementation_complete",
            "launch independent control/end_only/gated A/B after remote worktree integration test",
        ),
        (
            "a02-boss-order-ab", "code_ready_tests_passed",
            "keep main chain unchanged; stage independent treatment with penalty 0.09 and preference enabled",
        ),
        (
            "a08-evolve-shaping-ab", "code_ready_tests_passed",
            "keep main chain unchanged; stage three independent arms",
        ),
    ):
        target = items[experiment_id]
        target["status"] = status
        target["receipt"] = {
            "observedAt": observed,
            "tacticalShapingRevision": 4,
            "focusedTests": "10/10 combined pass locally and in remote temp runtime",
            "deployedToMain": False,
        }
        target["nextAction"] = next_action

    plan["updatedAt"] = observed
    plan["changeReason"] = (
        "All four PPO chains advanced without failed updates; all three frozen40 BC matrices completed; "
        "standard direct round1 passed 95-65; both candidates' two-round direct gates and a fresh four-line "
        "fixed Arena were launched; a guarded frozen40 round2 watcher was started; tactical revision 4 passed "
        "10 focused tests locally and remotely."
    )
    write(PLAN, plan)


if __name__ == "__main__":
    main()
