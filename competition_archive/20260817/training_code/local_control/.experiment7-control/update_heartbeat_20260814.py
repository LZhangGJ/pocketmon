from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")
PLAN = ROOT / "control/gold-acceleration-plan.json"
LATEST = ROOT / "monitoring/gold-acceleration/latest.json"


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    observed = datetime.now(timezone.utc).isoformat()
    ppo = {
        "a02_grim_g247": [267, 620, 1270459, 20, 0],
        "a02_grim_g247_pokegear": [267, 620, 1299255, 20, 0],
        "a08_rabsca": [292, 618, 797787, 15, 0],
        "a08_maxbelt": [292, 630, 854406, 15, 0],
        "lucario_gold_exact": [4, 1424, 2350770, 117, 0],
        "universal_ppo_standard_1m": [4, 143, 224141, 4, 0],
        "universal_ppo_large_256x6": [0, 151, 238330, 0, 0],
    }
    snapshot = {
        "schemaVersion": 1,
        "observedAt": observed,
        "engine_seed_controlled": False,
        "ppo": {
            name: {
                "generation": row[0],
                "completedShards": row[1],
                "decisions": row[2],
                "publishedUpdates": row[3],
                "failedUpdates": row[4],
            }
            for name, row in ppo.items()
        },
        "bcDirectGate": {
            "standard_1m": {
                "rounds": [0.59375, 0.56875],
                "combinedGames": 320,
                "combinedWins": 186,
                "combinedScoreRate": 0.58125,
            },
            "large_256x6": {
                "rounds": [0.525, 0.6125],
                "combinedGames": 320,
                "combinedWins": 182,
                "combinedScoreRate": 0.56875,
                "a02BaselineCombinedScoreRate": 0.475,
            },
        },
        "activeIssue": {
            "id": "universal-large-g1-rpc-wait",
            "status": "node_local_runtime_copying",
            "host": "doraemon20",
            "parentPid": 2140839,
            "childPid": 2141125,
            "evidence": "child remained D/rpc_wait for >55 minutes; GPU utilization 0%",
            "repair": "exact inputs verified locally; stream Python/Torch runtime from doraemon02, run once, atomically publish",
        },
        "tierAEvaluation": {
            "status": "controller-running-free-host-dispatch",
            "tierADecks": 45,
            "profiles": ["standard_1m", "large_256x6"],
            "fastScreenGamesPerCell": 4,
            "output": str(ROOT / "monitoring/ppo-vs-bc-tier-a"),
            "roundKey": "ab7439558336a6a6",
            "scheduledGames": 2520,
            "observedRows": 2070,
            "uniqueKeys": 2070,
            "failures": 16,
            "recovery": "running_failed_and_missing_keys_only",
        },
        "replay": {
            "latestWindowEnd": "2026-08-12",
            "nextMandatoryPublicationCheck": "2026-08-14T11:00:00+09:00",
        },
    }
    atomic_write(LATEST, snapshot)

    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    items = plan.setdefault("items", [])
    by_id = {item.get("experimentId"): item for item in items}
    updates = {
        "seven-chain-training": {
            "experimentId": "seven-chain-training",
            "objective": "Continuously train all seven active PPO chains",
            "hypothesis": "Distributed unique on-policy shards improve experts and Universal policies",
            "dependencies": [],
            "priority": "P1",
            "status": "running_with_large_rpc_repair",
            "owner": "main-controller",
            "allocation": "all reachable doraemon rollout workers; one learner per profile",
            "startCommand": "existing deduplicated seven-chain launchers",
            "startedAt": "2026-08-13T19:39:54+00:00",
            "successMetric": "generation and unique shard growth with failedUpdates flat",
            "stopCondition": "two samples without shard/generation growth and no active worker",
            "artifact": str(ROOT / "learners"),
            "receipt": snapshot["ppo"],
            "nextAction": "finish Universal large g1 node-local rescue",
        },
        "ppo-vs-bc-tier-a": {
            "experimentId": "ppo-vs-bc-tier-a",
            "objective": "Measure current PPO win rate against latest BC driving all Tier-A decks",
            "hypothesis": "Tier-A macro evaluation exposes transferable strength and weak archetypes",
            "dependencies": ["dual-bc-admission"],
            "priority": "P1",
            "status": "preparing_first_round",
            "owner": "arena-controller",
            "allocation": "all reachable idle doraemon hosts",
            "startCommand": "tier-A cached paired-seat controller",
            "startedAt": observed,
            "successMetric": "4-game screen for every PPO/deck/profile, weak cells expanded to 40",
            "stopCondition": "freeze each versioned round; never mix checkpoint versions",
            "artifact": str(ROOT / "monitoring/ppo-vs-bc-tier-a"),
            "receipt": snapshot["tierAEvaluation"],
            "nextAction": "launch first cached distributed quick screen",
        },
        "bc-direct-two-round": {
            "experimentId": "bc-direct-two-round",
            "objective": "Complete two independent same-deck new-vs-old BC rounds",
            "hypothesis": "Both admitted BCs outperform the prior Universal BC",
            "dependencies": ["dual-bc-admission"],
            "priority": "P1",
            "status": "complete",
            "owner": "bc-screening-controller",
            "allocation": "distributed guarded Arena",
            "startCommand": "/homes/lzhang/run_bc_direct_replacement_gate_20260814.py",
            "startedAt": "2026-08-13T19:13:00+00:00",
            "successMetric": "two 160-game paired-seat rounds per profile",
            "stopCondition": "complete after 320 unique games/profile",
            "artifact": "/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/0812-d14-ram-npz-fast-20260813/replacement-screening/direct-new-old",
            "receipt": snapshot["bcDirectGate"],
            "nextAction": "retain both admitted versions; target large A02-baseline weakness",
        },
    }
    for experiment_id, value in updates.items():
        if experiment_id in by_id:
            by_id[experiment_id].update(value)
        else:
            items.append(value)
    plan["updatedAt"] = observed
    atomic_write(PLAN, plan)


if __name__ == "__main__":
    main()
