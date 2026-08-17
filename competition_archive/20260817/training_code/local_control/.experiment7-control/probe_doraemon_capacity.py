from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path


def cpu_ticks() -> tuple[int, int]:
    fields = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
    return sum(fields), idle


total0, idle0 = cpu_ticks()
time.sleep(0.5)
total1, idle1 = cpu_ticks()
cpu_percent = 100.0 * (1.0 - (idle1 - idle0) / max(total1 - total0, 1))

memory = {}
for line in Path("/proc/meminfo").read_text().splitlines():
    key, value = line.split(":", 1)
    memory[key] = int(value.strip().split()[0])

io_avg10 = None
pressure = Path("/proc/pressure/io")
if pressure.is_file():
    first = pressure.read_text().splitlines()[0].split()
    for item in first:
        if item.startswith("avg10="):
            io_avg10 = float(item.split("=", 1)[1])

processes = {"arenaMatches": 0, "ppoRolloutOrLearner": 0, "otherPython": 0}
for cmdline_path in Path("/proc").glob("[0-9]*/cmdline"):
    try:
        command = cmdline_path.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except (OSError, PermissionError):
        continue
    if "run_local_match.py" in command:
        processes["arenaMatches"] += 1
    elif any(token in command for token in ("run_async_ppo_", "collect_universal_ppo_rollouts.py")):
        processes["ppoRolloutOrLearner"] += 1
    elif "python" in command:
        processes["otherPython"] += 1

candidate_mounts = [
    path
    for path in ("/suedata", "/suedata1", "/dataT0")
    if Path(path).is_dir()
]
required = {
    "python": Path("/homes/lzhang/mypath/new/envs/trans/bin/python3.11").is_file(),
    "worktree": Path("/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0").is_dir(),
    "loadGuard": Path("/homes/lzhang/run_load_guarded_arena_shard.sh").is_file(),
    "isolatedShard": Path("/homes/lzhang/run_isolated_arena_shard.sh").is_file(),
    "league": Path(
        "/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/"
        "state/league.json"
    ).is_file(),
    "cg": Path(
        "/dataT0/Free/lzhang/pocketmon-runs/experiment7-opponent-pool-20260810/"
        "official-engine/cg"
    ).is_dir(),
}

print(
    json.dumps(
        {
            "host": socket.gethostname(),
            "cpuCount": os.cpu_count(),
            "cpuPercent": round(cpu_percent, 1),
            "load1": round(os.getloadavg()[0], 2),
            "load5": round(os.getloadavg()[1], 2),
            "ioPressureAvg10": io_avg10,
            "memoryAvailableGiB": round(memory.get("MemAvailable", 0) / 1024 / 1024, 1),
            "candidateMounts": candidate_mounts,
            "required": required,
            "processes": processes,
        },
        ensure_ascii=False,
    )
)
