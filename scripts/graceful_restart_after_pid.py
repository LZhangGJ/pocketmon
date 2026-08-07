from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_matches(pid: int, expected: str) -> bool:
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except FileNotFoundError:
        return False
    return expected in command


def validate_stop_file(path: Path) -> Path:
    path = path.resolve()
    if not path.is_absolute() or path.name != "STOP":
        raise ValueError("stop file must be an absolute path named STOP")
    return path


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Restart a coordinator only after its exact PID exits")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--timeout-hours", type=float, default=4.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ValueError("restart command is required after --")
    stop_file = validate_stop_file(args.stop_file)
    if not stop_file.is_file():
        raise FileNotFoundError(f"graceful STOP marker is missing: {stop_file}")
    if not process_matches(args.pid, args.expected):
        raise RuntimeError("target PID is absent or does not match the expected coordinator")

    deadline = time.monotonic() + args.timeout_hours * 3600.0
    while process_matches(args.pid, args.expected):
        if time.monotonic() >= deadline:
            raise TimeoutError("target coordinator did not exit before the restart timeout")
        time.sleep(args.poll_seconds)
    if not stop_file.is_file():
        raise RuntimeError("STOP marker disappeared before the coordinator exited")
    stop_file.unlink()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_handle = args.log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=args.cwd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    report = {
        "completed_at": now(),
        "old_pid": args.pid,
        "new_pid": process.pid,
        "stop_file_removed": str(stop_file),
        "cwd": str(args.cwd),
        "command": command,
        "log": str(args.log),
    }
    atomic_json(args.report, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
