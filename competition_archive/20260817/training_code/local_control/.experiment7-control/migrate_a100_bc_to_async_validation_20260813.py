#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


PYTHON = "/homes/lzhang/mypath/new/envs/trans/bin/python"
CONTROLLER = "/homes/lzhang/run_async_bc_profile_controller_20260813.py"
OLD = Path("/tmp/lzhang-bc-capacity-a100-20260813")
NEW = Path("/tmp/lzhang-bc-capacity-a100-async-20260813")
RECEIPT = Path(
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/"
    "capacity-comparison-a100-ram-prefetch-b256/async-validation-migration.json"
)
PROFILES = {
    "standard_1m": {"trainGpu": "3", "validationGpu": "1"},
    "large_256x6": {"trainGpu": "7", "validationGpu": "4"},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def processes() -> dict[int, str]:
    found = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        found[int(entry.name)] = raw
    return found


def main() -> None:
    if not Path(CONTROLLER).is_file():
        raise FileNotFoundError(CONTROLLER)
    starts = {}
    for profile in PROFILES:
        old_root = OLD / f"{profile}-persistent-continuation"
        report = json.loads((old_root / "training_report.json").read_text())
        completed = [int(row["epoch"]) for row in report.get("epochs", [])]
        if not completed:
            raise RuntimeError(f"no validated epoch for {profile}")
        starts[profile] = max(completed) + 1
        for required in (old_root / "best_model.pt", old_root / "training_report.json"):
            if not required.is_file():
                raise FileNotFoundError(required)

    targets = {}
    for pid, command in processes().items():
        if (
            "train_universal_bc.py" in command
            and str(OLD) in command
        ) or "continue_a100_bc_persistent_20260813.py" in command:
            targets[pid] = command
    for pid in sorted(targets):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 30
    while time.time() < deadline:
        alive = [pid for pid in targets if Path(f"/proc/{pid}").exists()]
        if not alive:
            break
        time.sleep(0.5)
    alive = [pid for pid in targets if Path(f"/proc/{pid}").exists()]
    if alive:
        raise RuntimeError(f"old BC processes did not terminate: {alive}")

    launched = {}
    NEW.mkdir(parents=True, exist_ok=True)
    for profile, gpu in PROFILES.items():
        old_root = OLD / f"{profile}-persistent-continuation"
        command = [
            PYTHON, "-s", CONTROLLER, "--profile", profile,
            "--train-gpu", gpu["trainGpu"], "--validation-gpu", gpu["validationGpu"],
            "--start-epoch", str(starts[profile]), "--previous-root", str(old_root),
            "--output-root", str(NEW),
        ]
        log_path = NEW / f"controller-{profile}.log"
        log = log_path.open("a")
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
        )
        launched[profile] = {
            "controllerPid": process.pid, "startEpoch": starts[profile],
            **gpu, "log": str(log_path),
        }
    write(
        RECEIPT,
        {
            "schemaVersion": 1,
            "status": "migrated_to_persistent_training_with_async_validation",
            "migratedAt": now(),
            "terminatedProcesses": targets,
            "profiles": launched,
            "maxPendingValidations": 2,
            "dataReloadPolicy": "one load per persistent trainer and validator",
        },
    )
    print(json.dumps({"status": "migrated", "receipt": str(RECEIPT), "profiles": launched}))


if __name__ == "__main__":
    main()
