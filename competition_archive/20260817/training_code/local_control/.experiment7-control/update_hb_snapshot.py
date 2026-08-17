#!/usr/bin/env python3
import datetime as dt
import json
import os
import pathlib

ROOT = pathlib.Path(
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811"
)
SUMMARY = json.loads(
    pathlib.Path(os.environ.get("HB_SUMMARY", "/dev/shm/hb-summary.json")).read_text()
)
NOW = dt.datetime.now(dt.timezone.utc).isoformat()


def atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


names = [
    "a02_grim_g247",
    "a02_grim_g247_pokegear",
    "a08_rabsca",
    "a08_maxbelt",
    "lucario_gold_exact",
    "universal_ppo_standard_1m",
    "universal_ppo_large_256x6",
]
ppo = {}
for name in names:
    row = SUMMARY["chains"][name]
    ppo[name] = {
        key: row.get(key)
        for key in (
            "generation",
            "snapshotId",
            "completedShards",
            "episodes",
            "decisions",
            "externalWins",
            "externalLosses",
            "publishedUpdates",
            "failedUpdates",
        )
    }

latest = {
    "schemaVersion": 1,
    "observedAt": NOW,
    "engine_seed_controlled": False,
    "ppo": ppo,
    "frozenMatrix": {
        "roundId": "20260813T230421Z-a02_grim_g247-g000271-a02_grim_g247_pokegear-g000271-a08_maxbelt-g000295-a08_rabsca-g000295-lucario_gold_exact-g000007-universal_ppo_large_256x6-g000000-universal_ppo_standard_1m-g000007",
        "status": "complete",
        "games": 2576,
        "frozenAgentCount": 36,
        "completedAt": "2026-08-13T23:24:43.489597+00:00",
    },
    "tierA": {
        "roundKey": "ab7439558336a6a6",
        "status": "complete",
        "games": 2520,
        "failures": 0,
        "completedAt": "2026-08-13T23:14:10.385899+00:00",
    },
    "largePpo": {
        "publishedGeneration": ppo["universal_ppo_large_256x6"]["generation"],
        "publishedSnapshot": ppo["universal_ppo_large_256x6"]["snapshotId"],
        "generation5": {
            "host": "doraemon13",
            "parentPid": 5847,
            "trainerPid": None,
            "packagerPid": 7141,
            "state": "training_complete_packaging_rpc_wait_under_10_minutes",
            "gpuUtilizationPercent": 0,
            "gpuMemoryMiB": 1,
            "checkpointSha256": "e5c5f028dded8c66d9044a34354b88d31561df7b002c0b947b6abe1cd8a662cf",
            "approximateKl": 0.0005049831980431918,
            "clipFraction": 0.017743055634200575,
        },
    },
    "dedicatedTestServer": {
        "host": "doraemon16",
        "status": "dedicated_testing_only",
        "logicalCpus": 80,
        "cpuAggregatePercent": 0.4,
        "ioPressureAvg10Percent": 0.0,
        "normalCpuLimitPercent": 90,
        "urgentBurstPercent": 98,
        "misallocatedProjectProcesses": 0,
    },
    "replay": {
        "latestCompleteWindow": "2026-08-12",
        "official20260813Check": "inconclusive_remote_cli_rpc_wait_terminated",
        "nextAction": "retry from node-local Kaggle runtime; do not duplicate dates",
    },
    "repairs": [
        "universal large advanced from g1 to g4 after recovery",
        "large g5 training completed; package export entered short RPC wait and remains under two-sample observation",
        "doraemon16 rollout workers drained; server remains test-only",
        "specialist loss archive capped at 100 external losses per chain across newest three chronological generations",
    ],
}
loss_archive = ROOT / "monitoring/specialist-loss-replays/latest.json"
loss_controller = ROOT / "monitoring/specialist-loss-replays/controller-state.json"
if loss_archive.exists():
    latest["specialistLossArchive"] = json.loads(loss_archive.read_text())
if loss_controller.exists():
    latest["specialistLossArchiveController"] = json.loads(loss_controller.read_text())
atomic(ROOT / "monitoring/gold-acceleration/latest.json", latest)

plan_path = ROOT / "control/gold-acceleration-plan.json"
plan = json.loads(plan_path.read_text())
by_id = {item.get("experimentId"): item for item in plan.get("items", [])}
updates = {
    "ppo-active-four": {
        "status": "running_healthy",
        "receipt": {"observedAt": NOW, "chains": {name: ppo[name] for name in names[:4]}},
        "nextAction": "continue rollout; evaluate g272/g296 after node-local test packages are ready",
    },
}
for experiment_id, fields in updates.items():
    if experiment_id in by_id:
        by_id[experiment_id].update(fields)

for item in (
    {
        "experimentId": "universal-large-ppo-evolution",
        "objective": "Continuously evolve large Universal PPO over the 133-deck pool",
        "hypothesis": "Large Universal PPO improves broad policy coverage",
        "dependencies": ["large-bc-admitted"],
        "priority": "P1",
        "status": "generation5_trained_packaging_rpc_wait_observing",
        "owner": "large-ppo-controller",
        "allocation": "doraemon13 GPU0",
        "startCommand": "single persistent learner",
        "successMetric": "published generations grow with stable KL/clip and fixed Arena improves",
        "stopCondition": "only two-sample real stall or safety guard",
        "artifact": str(ROOT / "learners/universal_ppo_large_256x6"),
        "receipt": latest["largePpo"],
        "nextAction": "take a second 10-minute package sample; localize exact g5 export only if still rpc-wait without artifact growth",
    },
    {
        "experimentId": "doraemon16-dedicated-testing",
        "objective": "Keep one 80-core server exclusively for evidence-generating tests",
        "hypothesis": "Dedicated local-cache Arena removes scheduling contention",
        "dependencies": [],
        "priority": "P1",
        "status": "dedicated_testing_only_local_cache_pending",
        "owner": "main-controller",
        "allocation": "doraemon16 CPU; normal 90%, urgent receipt-bound 98%",
        "startCommand": "/homes/lzhang/run_load_guarded_arena_shard.sh",
        "successMetric": "latest checkpoints evaluated without rollout/training on host",
        "stopCondition": "never repurpose without user authorization",
        "artifact": str(ROOT / "control/server-reservations/doraemon16.json"),
        "receipt": latest["dedicatedTestServer"],
        "nextAction": "localize exact packages and engine assets, then rerun latest A02/Pokegear/MaxBelt",
    },
):
    if item["experimentId"] in by_id:
        by_id[item["experimentId"]].update(item)
    else:
        plan.setdefault("items", []).append(item)
atomic(plan_path, plan)
