from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import Experiment7Error, read_json, utc_now, write_json


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def validate_job(job: dict[str, Any]) -> None:
    required = {"jobId", "command", "runDir", "logPath", "receiptPath"}
    missing = required - set(job)
    if missing:
        raise Experiment7Error(f"job missing fields: {sorted(missing)}")
    command = job["command"]
    if not isinstance(command, list) or not command or not all(isinstance(value, str) and value for value in command):
        raise Experiment7Error("job command must be a non-empty string array")
    if "gpuIndex" in job and int(job["gpuIndex"]) < 0:
        raise Experiment7Error("gpuIndex must be non-negative")


def supervise(job_path: Path) -> int:
    job = read_json(job_path)
    validate_job(job)
    run_dir = Path(job["runDir"])
    log_path = Path(job["logPath"])
    receipt_path = Path(job["receiptPath"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    gpu = job.get("gpuIndex")
    lock_name = f"/tmp/pocketmon-experiment7-gpu-{gpu}.lock" if gpu is not None else "/tmp/pocketmon-experiment7-cpu.lock"
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in job.get("env", {}).items()})
    environment.update(
        {
            "PYTHONHASHSEED": str(job.get("pythonHashSeed", 0)),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    started_at = utc_now()
    with Path(lock_name).open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            payload = {
                "schemaVersion": 1,
                "jobId": job["jobId"],
                "status": "blocked_gpu_lock",
                "host": os.uname().nodename,
                "gpuIndex": gpu,
                "lock": lock_name,
                "startedAt": started_at,
                "finishedAt": utc_now(),
                "exitCode": 73,
            }
            write_json(receipt_path, payload)
            return 73
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(json.dumps({"jobId": job["jobId"], "pid": os.getpid(), "host": os.uname().nodename}))
        lock_handle.flush()
        with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
            log_handle.write(json.dumps({"event": "start", "job": job, "startedAt": started_at}, ensure_ascii=False) + "\n")
            process = subprocess.Popen(
                job["command"],
                cwd=job.get("cwd") or None,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running = {
                "schemaVersion": 1,
                "jobId": job["jobId"],
                "stage": job.get("stage"),
                "status": "running",
                "host": os.uname().nodename,
                "gpuIndex": gpu,
                "supervisorPid": os.getpid(),
                "childPid": process.pid,
                "startedAt": started_at,
                "command": job["command"],
                "cwd": job.get("cwd"),
                "logPath": str(log_path),
                "commit": job.get("commit"),
            }
            write_json(receipt_path, running)
            exit_code = process.wait()
            finished = {**running, "status": "succeeded" if exit_code == 0 else "failed", "finishedAt": utc_now(), "exitCode": exit_code}
            write_json(receipt_path, finished)
            log_handle.write(json.dumps({"event": "finish", "exitCode": exit_code, "finishedAt": finished["finishedAt"]}) + "\n")
            return exit_code


def launch(job_path: Path) -> dict[str, Any]:
    job = read_json(job_path)
    validate_job(job)
    receipt_path = Path(job["receiptPath"])
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt.get("status") == "running" and _pid_alive(int(receipt.get("supervisorPid", 0))):
            raise Experiment7Error(f"job is already running: {job['jobId']}")
    supervisor_log = Path(job["runDir"]) / "supervisor.log"
    supervisor_log.parent.mkdir(parents=True, exist_ok=True)
    with supervisor_log.open("ab") as handle:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "supervise", "--job", str(job_path.resolve())],
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    payload = {
        "schemaVersion": 1,
        "jobId": job["jobId"],
        "status": "launched",
        "host": os.uname().nodename,
        "supervisorPid": process.pid,
        "jobPath": str(job_path.resolve()),
        "receiptPath": str(receipt_path),
        "supervisorLog": str(supervisor_log),
        "launchedAt": utc_now(),
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def status(job_path: Path) -> dict[str, Any]:
    job = read_json(job_path)
    receipt_path = Path(job["receiptPath"])
    if not receipt_path.exists():
        payload = {"jobId": job["jobId"], "status": "not_started", "receiptPath": str(receipt_path)}
    else:
        payload = read_json(receipt_path)
        if payload.get("status") == "running":
            payload["supervisorAlive"] = _pid_alive(int(payload.get("supervisorPid", 0)))
            payload["childAlive"] = _pid_alive(int(payload.get("childPid", 0)))
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Detached, lock-protected Linux worker for Experiment 7 jobs")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("launch", "supervise", "status"):
        child = sub.add_parser(name)
        child.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "launch":
        launch(args.job.resolve())
    elif args.command == "supervise":
        raise SystemExit(supervise(args.job.resolve()))
    else:
        status(args.job.resolve())


if __name__ == "__main__":
    main()
