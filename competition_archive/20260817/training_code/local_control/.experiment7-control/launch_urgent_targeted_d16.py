#!/usr/bin/env python3
import csv
import datetime as dt
import json
import os
import pathlib
import random
import subprocess

ROOT = pathlib.Path(
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811"
)
MATRIX_ROUND = ROOT / "monitoring/full-matrix/rounds/20260813T230421Z-a02_grim_g247-g000271-a02_grim_g247_pokegear-g000271-a08_maxbelt-g000295-a08_rabsca-g000295-lucario_gold_exact-g000007-universal_ppo_large_256x6-g000000-universal_ppo_standard_1m-g000007"
EVAL_ROOT = ROOT / "monitoring/urgent-targeted-compare"
ROUND_ID = "20260814T0836JST-g271-g271-g295-attempt2"
ROUND = EVAL_ROOT / "rounds" / ROUND_ID
WORKTREE = "/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0"
PYTHON = "/homes/lzhang/mypath/new/envs/trans/bin/python3.11"
CG_DIR = "/dataT0/Free/lzhang/pocketmon-runs/experiment7-opponent-pool-20260810/official-engine/cg"
SHARDS = 12

A02 = "live_a02_grim_g247_g000271__a02_g247_baseline"
POKE = "live_a02_grim_g247_pokegear_g000271__a02_g247_pokegear"
BELT = "live_a08_maxbelt_g000295__a08_maxbelt"
LARGE_A02 = "universal_bc_20260812_large_256x6__universal_bc_baseline__a02_g247_baseline"
LARGE_POKE = "universal_bc_20260812_large_256x6__universal_bc_baseline__a02_g247_pokegear"
LARGE_BELT = "universal_bc_20260812_large_256x6__universal_bc_baseline__a08_maxbelt"
MATCHES = [
    (POKE, A02, "a02_pokegear_vs_a02"),
    (BELT, A02, "a08_maxbelt_vs_a02"),
    (BELT, POKE, "a08_maxbelt_vs_a02_pokegear"),
    (A02, LARGE_A02, "a02_vs_large_bc_same_deck"),
    (POKE, LARGE_POKE, "a02_pokegear_vs_large_bc_same_deck"),
    (BELT, LARGE_BELT, "a08_maxbelt_vs_large_bc_same_deck"),
]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


ROUND.mkdir(parents=True, exist_ok=True)
(ROUND / "raw").mkdir(exist_ok=True)
(ROUND / "logs").mkdir(exist_ok=True)
schedule = []
for match_index, (learner, opponent, label) in enumerate(MATCHES):
    for seat in (0, 1):
        for game_index in range(10):
            schedule.append(
                {
                    "learner": learner,
                    "opponent": opponent,
                    "seed": 814_000_000 + match_index * 10_000 + seat * 1_000 + game_index,
                    "learner_seat": seat,
                    "label": label,
                }
            )
random.Random(8140830).shuffle(schedule)
schedule_path = ROUND / "schedule.csv"
with schedule_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle, fieldnames=("learner", "opponent", "seed", "learner_seat")
    )
    writer.writeheader()
    writer.writerows(
        {key: row[key] for key in writer.fieldnames} for row in schedule
    )

receipt = {
    "schemaVersion": 1,
    "roundId": ROUND_ID,
    "host": "doraemon16",
    "status": "running",
    "startedAt": now(),
    "wholeMachineCpuLimitPercent": 98,
    "normalCpuLimitPercent": 90,
    "ioPressureLimitPercent": 80,
    "engineSeedControlled": False,
    "games": len(schedule),
    "shards": SHARDS,
    "matches": [label for _, _, label in MATCHES],
}
atomic_json(ROUND / "receipt.json", receipt)
atomic_json(EVAL_ROOT / "active.json", receipt)

env = dict(os.environ)
env["ARENA_CPU_LIMIT_PERCENT"] = "98"
env["ARENA_IO_LIMIT_PERCENT"] = "80"
processes = []
for shard in range(SHARDS):
    output = ROUND / "raw" / f"results-shard-{shard:03d}.csv"
    log = (ROUND / "logs" / f"shard-{shard:03d}.log").open("w")
    command = [
        "/homes/lzhang/run_load_guarded_arena_shard.sh",
        WORKTREE,
        PYTHON,
        str(schedule_path),
        str(MATRIX_ROUND / "learners.json"),
        str(MATRIX_ROUND / "opponents.json"),
        CG_DIR,
        str(output),
        str(shard),
        str(SHARDS),
        str(ROUND),
    ]
    processes.append((shard, subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))

return_codes = {}
for shard, process, log in processes:
    return_codes[str(shard)] = process.wait()
    log.close()

rows = []
for path in sorted((ROUND / "raw").glob("results-shard-*.csv")):
    with path.open(newline="", encoding="utf-8") as handle:
        rows.extend(csv.DictReader(handle))

summary = {}
for learner, opponent, label in MATCHES:
    selected = [row for row in rows if row["learner"] == learner and row["opponent"] == opponent]
    summary[label] = {
        "games": len(selected),
        "wins": sum(row["result"] == "win" for row in selected),
        "losses": sum(row["result"] == "loss" for row in selected),
        "draws": sum(row["result"] == "draw" for row in selected),
        "failures": sum(bool(row.get("failure")) for row in selected),
        "seat0": {
            "games": sum(row["learner_seat"] == "0" for row in selected),
            "wins": sum(row["learner_seat"] == "0" and row["result"] == "win" for row in selected),
        },
        "seat1": {
            "games": sum(row["learner_seat"] == "1" for row in selected),
            "wins": sum(row["learner_seat"] == "1" and row["result"] == "win" for row in selected),
        },
    }

report = {
    **receipt,
    "status": "complete",
    "completedAt": now(),
    "resultRows": len(rows),
    "returnCodes": return_codes,
    "summary": summary,
}
atomic_json(ROUND / "report.json", report)
atomic_json(EVAL_ROOT / "latest.json", report)
atomic_json(EVAL_ROOT / "active.json", report)
