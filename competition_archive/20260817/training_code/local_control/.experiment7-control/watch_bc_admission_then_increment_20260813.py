#!/usr/bin/env python3
"""Start the next TensorDict BC window immediately after the 8/11 BC is admitted."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


CURRENT = Path("/tmp/lzhang-bc-capacity-a100-20260813")
SHARED_CURRENT = Path(
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/"
    "capacity-comparison-a100-ram-prefetch-b256"
)
DAILY = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc")
WINDOW = Path(
    "/suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/"
    "windows/2026-08-12/tensordict-sources.json"
)
RUNNER = Path("/homes/lzhang/run_incremental_a100_bc_profile_20260813.py")
OUTPUT = DAILY / "training"
ADMISSION = DAILY / "admissions/2026-08-11.json"
STATE = DAILY / "control/8-11-admission-to-8-12-increment.json"
PROFILE_GPUS = {
    "standard_1m": {"train": "3", "validation": "1"},
    "large_256x6": {"train": "7", "validation": "4"},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(payload: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_name(f".{STATE.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, STATE)


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def current_finished() -> bool:
    for profile in PROFILE_GPUS:
        state = CURRENT / f"{profile}-persistent-continuation/continuation_state.json"
        if not state.is_file() or read(state).get("status") != "complete":
            return False
    return True


def main() -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = (STATE.parent / "8-11-admission-to-8-12-increment.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("HANDOFF_WATCHER_ALREADY_RUNNING", flush=True)
        return

    while True:
        conditions = {
            "oldWindowTrainingComplete": current_finished(),
            "oldWindowAdmitted": ADMISSION.is_file() and read(ADMISSION).get("status") == "admitted",
            "newTensorDictWindowReady": WINDOW.is_file(),
            "runnerReady": RUNNER.is_file(),
        }
        write(
            {
                "schemaVersion": 1,
                "status": "waiting" if not all(conditions.values()) else "launching",
                "checkedAt": now(),
                "conditions": conditions,
                "admissionReceipt": str(ADMISSION),
                "nextSources": str(WINDOW),
            }
        )
        if all(conditions.values()):
            break
        time.sleep(15)

    admission = read(ADMISSION)
    pids = {}
    for profile, gpus in PROFILE_GPUS.items():
        current_root = CURRENT / f"{profile}-persistent-continuation"
        init = current_root / "selected_best_model.pt"
        baseline_report = current_root / "training_report.json"
        if not baseline_report.is_file():
            baseline_report = SHARED_CURRENT / profile / "training_report.json"
        if not init.is_file():
            raise FileNotFoundError(init)
        command = [
            "/homes/lzhang/mypath/new/envs/trans/bin/python", "-s", str(RUNNER),
            "--profile", profile, "--gpu", gpus["train"],
            "--validation-gpu", gpus["validation"],
            "--sources", str(WINDOW), "--initialize-from", str(init),
            "--baseline-report", str(baseline_report),
            "--output-root", str(OUTPUT), "--window-end", "2026-08-12",
        ]
        log_path = STATE.parent / f"incremental-{profile}-2026-08-12.log"
        log = log_path.open("a")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pids[profile] = {
            "pid": process.pid, "trainGpu": gpus["train"],
            "validationGpu": gpus["validation"], "log": str(log_path),
        }
    write(
        {
            "schemaVersion": 1,
            "status": "launched",
            "launchedAt": now(),
            "sourceAdmission": admission,
            "nextSources": str(WINDOW),
            "profiles": pids,
            "trainingMode": "persistent epochs with asynchronous checkpoint validation and early stopping",
        }
    )
    print(json.dumps({"status": "launched", "profiles": pids}), flush=True)


if __name__ == "__main__":
    main()
