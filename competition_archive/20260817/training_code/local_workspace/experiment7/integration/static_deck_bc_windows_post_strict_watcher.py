from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REMOTE_STATE = "/dataT0/Free/lzhang/pocketmon-runs/experiment7-static-deck-bc-10d-20260815/control/post-strict/state.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def acquire(path: Path) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "createdAt": now()}, handle)
        handle.write("\n")


def remote_gate() -> dict[str, Any] | None:
    result = subprocess.run(["ssh", "doraemon02", "cat", REMOTE_STATE], text=True, capture_output=True)
    if result.returncode:
        return None
    try:
        state = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    verification = state.get("strictVerification")
    if isinstance(verification, dict) and verification.get("parity") and len(verification.get("days", [])) == 10:
        return state
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()
    persistent = args.persistent_root.resolve()
    scratch = args.scratch_root.resolve()
    control = persistent / "control"
    state_path = control / "windows-post-strict-state.json"
    acquire(control / "windows-post-strict-controller.lock")
    state = {
        "schemaVersion": 1,
        "kind": "experiment7_static_deck_bc_windows_post_strict_watcher",
        "status": "waiting_shared_strict_10_of_10_success_and_parity",
        "pid": os.getpid(),
        "persistentRoot": str(persistent),
        "scratchRoot": str(scratch),
        "formalTrainingStarted": False,
        "createdAt": now(),
    }
    atomic_json(state_path, state)
    while True:
        gate = remote_gate()
        if gate is not None:
            break
        state["observedAt"] = now()
        atomic_json(state_path, state)
        time.sleep(args.poll_seconds)
    state.update({"status": "strict_verified_launching_windows_grim", "strictVerification": gate["strictVerification"], "observedAt": now()})
    atomic_json(state_path, state)
    launcher = persistent / "runtime" / "source" / "experiment7" / "integration" / "windows_static_deck_bc_launch.py"
    completed = subprocess.run([
        str(persistent / "runtime" / "venv" / "Scripts" / "python.exe"),
        str(launcher), "--persistent-root", str(persistent), "--scratch-root", str(scratch), "--state", str(state_path),
    ])
    if completed.returncode:
        atomic_json(state_path, {**state, "status": "windows_grim_launch_failed", "returnCode": completed.returncode, "observedAt": now()})
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
