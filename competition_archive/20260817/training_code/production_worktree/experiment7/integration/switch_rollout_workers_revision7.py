from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cmdline(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [token.decode("utf-8") for token in raw.split(b"\0") if token]


def worker_processes(worker_script: Path) -> list[tuple[int, list[str]]]:
    result = []
    self_pid = os.getpid()
    target = str(worker_script)
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == self_pid:
            continue
        try:
            argv = cmdline(int(proc.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeError):
            continue
        if target in argv and any("python" in Path(token).name for token in argv[:2]):
            result.append((int(proc.name), argv))
    return sorted(result)


def children(pid: int) -> list[int]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return [int(token) for token in path.read_text().split()]
    except (FileNotFoundError, ProcessLookupError):
        return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Switch persistent rollout-worker parents to revision 7"
    )
    parser.add_argument("--worker-script", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--stop-only", action="store_true")
    args = parser.parse_args()

    hostname = os.uname().nodename
    if "doraemon16" in hostname.lower() and not args.stop_only:
        raise RuntimeError("doraemon16 is arena_testing_only")
    worker_script = args.worker_script.resolve()
    actual_sha = sha256(worker_script)
    if actual_sha != args.expected_sha256:
        raise RuntimeError(f"worker SHA mismatch: {actual_sha}")

    lock_path = Path("/dev/shm/lzhang-experiment7-revision7-worker-switch.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        found = worker_processes(worker_script)
        switched = []
        log_root = args.receipt_root.resolve() / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        for old_pid, argv in found:
            old_children = children(old_pid)
            os.kill(old_pid, signal.SIGTERM)
            deadline = time.monotonic() + 10
            while Path(f"/proc/{old_pid}").exists() and time.monotonic() < deadline:
                time.sleep(0.1)
            if Path(f"/proc/{old_pid}").exists():
                raise RuntimeError(f"worker parent did not exit after SIGTERM: {old_pid}")
            new_pid = None
            log_path = None
            if not args.stop_only:
                worker_id = (
                    argv[argv.index("--worker-id") + 1]
                    if "--worker-id" in argv
                    else f"worker-{old_pid}"
                )
                log_path = log_root / f"{hostname}-{worker_id}.log"
                with log_path.open("ab") as log:
                    process = subprocess.Popen(
                        argv,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                new_pid = process.pid
            switched.append(
                {
                    "oldPid": old_pid,
                    "oldChildrenLeftToFinish": old_children,
                    "newPid": new_pid,
                    "argv": argv,
                    "log": str(log_path) if log_path else None,
                }
            )

        payload = {
            "schemaVersion": 1,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "hostname": hostname,
            "workerScript": str(worker_script),
            "workerSha256": actual_sha,
            "mode": "stop_only" if args.stop_only else "revision7_restart",
            "workers": switched,
        }
        receipt_root = args.receipt_root.resolve()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        atomic_write(receipt_root / f"{hostname}-{stamp}.json", payload)
        atomic_write(receipt_root / f"{hostname}-latest.json", payload)
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
