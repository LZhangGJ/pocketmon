from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


MAIN = Path(
    "/dataT0/Free/lzhang/pocketmon-runs/"
    "experiment7-async-ppo-league-20260811"
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    observed_at = now()
    league = read(MAIN / "state/league.json")
    pool = read(MAIN / "state/opponent-pool-live.json")
    promotion = read(MAIN / "control/ppo-frozen-promotion-state.json")
    chain = league["chains"]["universal_ppo_large_256x6"]
    cohort_path = Path(chain["current"]["deckCohortReceipt"])
    cohort = read(cohort_path)
    agents = pool["agents"]
    universal_agents = [
        row for row in agents if row.get("ppo_chain") == "universal_ppo_large_256x6"
    ]
    frozen_agents = [
        row for row in agents if row.get("pool_status") == "frozen_ppo_version"
    ]
    evidence = {
        "observedAt": observed_at,
        "universalSnapshotId": chain["current"]["snapshotId"],
        "cohortReceipt": str(cohort_path),
        "cohortSize": cohort["size"],
        "tierCounts": cohort["tierCounts"],
        "tierProbabilities": cohort["tierProbabilities"],
        "livePoolAgents": len(agents),
        "universalLiveAgents": len(universal_agents),
        "staleFrozenAgents": len(frozen_agents),
        "promotionAdmissions": len(promotion.get("admissions", {})),
        "promotionCandidates": {
            key: {
                "generation": value.get("generation"),
                "snapshotId": value.get("snapshotId"),
                "status": value.get("status"),
            }
            for key, value in promotion.get("candidates", {}).items()
            if value.get("status") == "testing"
        },
        "promotionExcludesPoolStatus": "large_g9_frozen_133_deck",
    }

    receipt_path = (
        MAIN
        / "control/large-g9-pool-reconfiguration-20260814"
        / "cohort20-reconfiguration-receipt.json"
    )
    receipt = {
        "schemaVersion": 1,
        "kind": "universal-ppo-tiered-20-deck-cohort-reconfiguration",
        **evidence,
        "atomicPoolBehavior": (
            "old generation stays deployable until checkpoint, portable packages, "
            "cohort receipt, and league/pool publication complete under the league lock"
        ),
        "rolloutBoundaryPolicy": (
            "workers finish the current shard, restart with CPU limit 70, then read the "
            "immutable current generation cohort"
        ),
        "dedicatedTestNode": "doraemon16",
        "a100UsedForRollout": False,
    }
    atomic_write(receipt_path, receipt)

    plan_path = MAIN / "control/gold-acceleration-plan.json"
    plan = read(plan_path)
    experiment_id = "universal-tiered-cohort20-curated-pool-20260815"
    item = {
        "experimentId": experiment_id,
        "objective": (
            "Drive Universal Large PPO with a deterministic weighted 20-deck cohort per "
            "generation while keeping the curated live/frozen pool asynchronous"
        ),
        "hypothesis": (
            "Twenty rotating decks preserve A/B/C/D coverage and reduce packaging and "
            "per-generation variance versus exposing all 133 decks simultaneously"
        ),
        "dependencies": [
            "complete 133-deck catalog",
            "immutable Universal Large checkpoint/package",
            "atomic league/pool lock",
        ],
        "priority": "P1",
        "status": "running_verification",
        "owner": "main-controller",
        "allocation": (
            "all reachable rollout nodes except dedicated test node doraemon16; "
            "whole-machine CPU limit 70; CPU inference only"
        ),
        "startCommand": (
            "run_async_ppo_learner.py --learner-deck-cohort-size 20; "
            "run_async_ppo_rollout_worker.py --cpu-limit 70"
        ),
        "startedAt": observed_at,
        "successMetric": (
            "every new Universal shard records learnerDeckCount=20; every successful "
            "generation atomically replaces its prior 20 live packages; non-133 promotion "
            "gate passes two complete rounds"
        ),
        "stopCondition": (
            "cohort receipt/package parity fails, a live-pool switch is non-atomic, or "
            "whole-machine CPU exceeds 70 without an active drain"
        ),
        "artifact": str(cohort_path.parent),
        "receipt": str(receipt_path),
        "evidence": evidence,
        "nextAction": (
            "verify the first post-boundary Universal shard records 20 decks and retain the "
            "current non-133 frozen-promotion round"
        ),
    }
    items = plan.setdefault("items", [])
    for index, current in enumerate(items):
        if current.get("experimentId") == experiment_id:
            items[index] = item
            break
    else:
        items.append(item)
    plan["updatedAt"] = observed_at
    plan["changeReason"] = (
        "user corrected Universal per-generation sample size from ten to twenty decks"
    )
    plan.setdefault("changeLog", []).append(
        {
            "at": observed_at,
            "reason": plan["changeReason"],
            "receipt": str(receipt_path),
        }
    )
    atomic_write(plan_path, plan)

    latest_path = MAIN / "monitoring/gold-acceleration/latest.json"
    latest = read(latest_path)
    latest["updatedAt"] = observed_at
    latest["universalCohort20"] = evidence
    latest.setdefault("events", []).append(
        {
            "at": observed_at,
            "event": "UNIVERSAL_COHORT_SIZE_CORRECTED_TO_20",
            "receipt": str(receipt_path),
        }
    )
    atomic_write(latest_path, latest)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
