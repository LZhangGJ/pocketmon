#!/usr/bin/env python3
import datetime as dt
import json
import os
import pathlib
import signal
import time

ROOT = pathlib.Path(
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811"
)
RECEIPT = ROOT / "control" / "server-reservations" / "doraemon16.json"
WRAPPERS = {
    877742: "--worker-id doraemon16 ",
    879817: "--worker-id doraemon16-v7b ",
}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def cmdline(pid):
    try:
        return pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError):
        return ""


def children(pid):
    path = pathlib.Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return [int(value) for value in path.read_text().split()]
    except (FileNotFoundError, ProcessLookupError):
        return []


def state(pid):
    try:
        for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("State:"):
                return line.split()[1]
    except (FileNotFoundError, ProcessLookupError):
        pass
    return "missing"


def write_receipt(status, drained=None):
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": 1,
        "host": "doraemon16",
        "role": "arena_testing_only",
        "status": status,
        "cpuLimitPercent": 90,
        "urgentBurst": {
            "allowed": True,
            "ceilingPercent": 98,
            "maxMinutes": 30,
            "requiresReceipt": True,
            "restoreCpuLimitPercent": 90,
        },
        "ioPressureLimitPercent": 80,
        "allow": [
            "frozen_pool_matrix",
            "ppo_vs_bc_tier_a",
            "tactical_arena",
            "bc_replacement_screening",
        ],
        "deny": ["ppo_rollout", "ppo_learner", "bc_training"],
        "updatedAt": now(),
        "drainedWrappers": drained or [],
    }
    tmp = RECEIPT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, RECEIPT)


valid = []
for pid, marker in WRAPPERS.items():
    command = cmdline(pid)
    if command and "run_async_ppo_rollout_worker.py" in command and marker in command:
        os.kill(pid, signal.SIGSTOP)
        valid.append(pid)

write_receipt("draining_current_rollout_shards", [])

while True:
    active_collectors = []
    for wrapper in valid:
        for child in children(wrapper):
            command = cmdline(child)
            if "collect_universal_ppo_rollouts.py" in command and state(child) != "Z":
                active_collectors.append(child)
    if not active_collectors:
        break
    time.sleep(5)

for pid in valid:
    try:
        os.kill(pid, signal.SIGTERM)
        os.kill(pid, signal.SIGCONT)
    except ProcessLookupError:
        pass

deadline = time.time() + 30
while time.time() < deadline and any(cmdline(pid) for pid in valid):
    time.sleep(1)

write_receipt("dedicated_testing_only", valid)
