from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-pid", type=int, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--visible-gpu", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()

    lock_path = args.state.with_suffix(args.state.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        queue = args.output_dir / "validation-queue"
        write_json(
            args.state,
            {
                "schemaVersion": 1,
                "status": "waiting_for_first_checkpoint",
                "watcherPid": os.getpid(),
                "trainerPid": args.trainer_pid,
                "createdAt": now(),
            },
        )
        while True:
            jobs = sorted(queue.glob("epoch_*.json")) if queue.is_dir() else []
            if jobs:
                break
            if not alive(args.trainer_pid):
                write_json(
                    args.state,
                    {
                        "schemaVersion": 1,
                        "status": "trainer_exited_before_checkpoint",
                        "watcherPid": os.getpid(),
                        "trainerPid": args.trainer_pid,
                        "recordedAt": now(),
                    },
                )
                return 1
            time.sleep(args.poll_seconds)

        command = [
            "ionice",
            "-c2",
            "-n7",
            "nice",
            "-n",
            "10",
            str(args.python),
            "-s",
            str(args.validator),
            "--sources",
            str(args.sources),
            "--output-dir",
            str(args.output_dir),
            "--baseline-report",
            str(args.baseline_report),
            "--baseline-checkpoint",
            str(args.baseline_checkpoint),
            "--device",
            "cuda:0",
            "--batch-size",
            "256",
            "--patience",
            "3",
            "--min-semantic-delta",
            "0.001",
            "--max-brier-increase",
            "0.005",
        ]
        env = dict(os.environ)
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": args.visible_gpu,
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("ab") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            write_json(
                args.state,
                {
                    "schemaVersion": 1,
                    "status": "rescue_validator_running",
                    "watcherPid": os.getpid(),
                    "trainerPid": args.trainer_pid,
                    "validatorPid": process.pid,
                    "firstQueueJob": str(jobs[0]),
                    "startedAt": now(),
                    "command": command,
                },
            )
            code = process.wait()
        write_json(
            args.state,
            {
                "schemaVersion": 1,
                "status": "complete" if code == 0 else "failed",
                "watcherPid": os.getpid(),
                "trainerPid": args.trainer_pid,
                "validatorPid": process.pid,
                "returnCode": code,
                "completedAt": now(),
            },
        )
        return code


if __name__ == "__main__":
    raise SystemExit(main())
