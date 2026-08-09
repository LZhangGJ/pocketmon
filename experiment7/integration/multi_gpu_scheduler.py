from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from common import Experiment7Error, read_json, utc_now, write_csv, write_json


DEFAULT_HOSTS = ["doraemon02", "doraemon03", "doraemon15", "doraemon16", "doraemon19", "doraemon20"]


@dataclass(frozen=True)
class GPU:
    host: str
    index: int
    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    utilization: int

    @property
    def idle_score(self) -> tuple[int, int, int]:
        return (self.utilization, -self.free_mib, self.index)


def ssh(host: str, command: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", host, command],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )


def inventory(
    hosts: list[str],
    output: Path,
    minimum_free_mib: int,
    maximum_utilization: int,
    ssh_timeout_seconds: int = 60,
) -> dict[str, Any]:
    rows = []
    errors = []
    query = (
        "set -e; hostname; "
        "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu "
        "--format=csv,noheader,nounits"
    )
    for host in hosts:
        try:
            result = ssh(host, query, timeout=ssh_timeout_seconds)
        except subprocess.TimeoutExpired:
            errors.append({"host": host, "error": "ssh_timeout"})
            continue
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode != 0 or len(lines) < 2:
            errors.append({"host": host, "error": result.stdout[-1000:], "returnCode": result.returncode})
            continue
        remote_name = lines[0]
        for line in lines[1:]:
            values = [value.strip() for value in line.split(",")]
            if len(values) != 6:
                errors.append({"host": host, "error": f"unexpected nvidia-smi row: {line}"})
                continue
            index, name, total, used, free, utilization_value = values
            row = {
                "host": host,
                "remoteHost": remote_name,
                "gpuIndex": int(index),
                "name": name,
                "totalMiB": int(total),
                "usedMiB": int(used),
                "freeMiB": int(free),
                "utilizationPercent": int(utilization_value),
            }
            row["eligible"] = row["freeMiB"] >= minimum_free_mib and row["utilizationPercent"] <= maximum_utilization
            rows.append(row)
    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "minimumFreeMiB": minimum_free_mib,
        "maximumUtilizationPercent": maximum_utilization,
        "sshTimeoutSeconds": ssh_timeout_seconds,
        "gpus": rows,
        "errors": errors,
    }
    write_json(output, payload)
    write_csv(output.with_suffix(".csv"), rows)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def eligible_gpus(inventory_payload: dict[str, Any]) -> list[GPU]:
    values = [
        GPU(host=row["host"], index=int(row["gpuIndex"]), name=row["name"], total_mib=int(row["totalMiB"]), used_mib=int(row["usedMiB"]), free_mib=int(row["freeMiB"]), utilization=int(row["utilizationPercent"]))
        for row in inventory_payload["gpus"]
        if row.get("eligible")
    ]
    values.sort(key=lambda gpu: gpu.idle_score)
    return values


def make_training_plan(
    inventory_path: Path,
    output: Path,
    worktree: PurePosixPath,
    commit: str,
    python: str,
    sources: PurePosixPath,
    run_root: PurePosixPath,
    stage: str,
    pretrain_checkpoint: PurePosixPath | None,
    seeds: list[int],
) -> dict[str, Any]:
    gpus = eligible_gpus(read_json(inventory_path))
    required = 1 if stage in {"pretrain", "smoke"} else len(seeds)
    if len(gpus) < required:
        raise Experiment7Error(f"stage {stage} needs {required} idle GPUs, found {len(gpus)}")
    integration = worktree / "experiment7" / "integration"
    jobs = []
    if stage == "pretrain":
        assignments = [("pretrain", 20260808, gpus[0])]
    elif stage == "smoke":
        assignments = [("smoke", 20260808, gpus[0])]
    elif stage == "finetune":
        if pretrain_checkpoint is None:
            raise ValueError("finetune requires pretrain_checkpoint")
        assignments = [(f"finetune-seed-{seed}", seed, gpus[index]) for index, seed in enumerate(seeds)]
    else:
        raise ValueError(stage)
    for name, seed, gpu in assignments:
        run_dir = run_root / name
        if stage == "pretrain":
            command = [python, str(integration / "train_driver.py"), "pretrain", "--sources", str(sources), "--output-dir", str(run_dir), "--seed", str(seed)]
        elif stage == "smoke":
            command = [python, str(integration / "train_driver.py"), "smoke", "--sources", str(sources), "--output-dir", str(run_dir), "--seed", str(seed)]
        else:
            command = [python, str(integration / "train_driver.py"), "finetune", "--sources", str(sources), "--pretrain-checkpoint", str(pretrain_checkpoint), "--output-dir", str(run_dir), "--seed", str(seed)]
        job_path = run_root / "jobs" / f"{name}.json"
        jobs.append({"jobId": name, "stage": stage, "host": gpu.host, "gpuIndex": gpu.index, "commit": commit, "cwd": str(worktree), "command": command, "runDir": str(run_dir), "logPath": str(run_dir / "train.log"), "receiptPath": str(run_dir / "job_receipt.json"), "jobPath": str(job_path), "env": {"PYTHON": python}})
    payload = {"schemaVersion": 1, "createdAt": utc_now(), "stage": stage, "commit": commit, "worktree": str(worktree), "sources": str(sources), "jobs": jobs}
    write_json(output, payload)
    write_csv(output.with_suffix(".csv"), [{"job_id": row["jobId"], "stage": row["stage"], "host": row["host"], "gpu_index": row["gpuIndex"], "run_dir": row["runDir"], "receipt": row["receiptPath"], "command": json.dumps(row["command"], ensure_ascii=False)} for row in jobs])
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def _upload_json(host: str, remote_path: str, payload: dict[str, Any]) -> None:
    encoded = base64.b64encode(json.dumps(payload, ensure_ascii=False).encode("utf-8")).decode("ascii")
    command = f"mkdir -p {shell_quote(str(PurePosixPath(remote_path).parent))}; printf %s {shell_quote(encoded)} | base64 -d > {shell_quote(remote_path)}"
    result = ssh(host, command)
    if result.returncode != 0:
        raise Experiment7Error(f"failed to upload job to {host}: {result.stdout}")


def shell_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def launch_plan(plan_path: Path, remote_python: str, worker_path: PurePosixPath) -> dict[str, Any]:
    plan = read_json(plan_path)
    launched = []
    for job in plan["jobs"]:
        _upload_json(job["host"], job["jobPath"], job)
        command = f"{shell_quote(remote_python)} {shell_quote(str(worker_path))} launch --job {shell_quote(job['jobPath'])}"
        result = ssh(job["host"], command)
        if result.returncode != 0:
            raise Experiment7Error(f"launch failed on {job['host']}: {result.stdout}")
        try:
            receipt = json.loads(result.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise Experiment7Error(f"invalid launch receipt from {job['host']}: {result.stdout}") from exc
        launched.append(receipt)
    payload = {"schemaVersion": 1, "createdAt": utc_now(), "plan": str(plan_path.resolve()), "launched": launched}
    output = plan_path.with_name(plan_path.stem + "_launch_receipt.json")
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def status_plan(plan_path: Path, remote_python: str, worker_path: PurePosixPath) -> dict[str, Any]:
    plan = read_json(plan_path)
    statuses = []
    for job in plan["jobs"]:
        command = f"{shell_quote(remote_python)} {shell_quote(str(worker_path))} status --job {shell_quote(job['jobPath'])}"
        result = ssh(job["host"], command)
        if result.returncode != 0:
            statuses.append({"jobId": job["jobId"], "host": job["host"], "status": "ssh_error", "output": result.stdout[-1000:]})
            continue
        statuses.append(json.loads(result.stdout.strip().splitlines()[-1]))
    payload = {"schemaVersion": 1, "createdAt": utc_now(), "plan": str(plan_path.resolve()), "statuses": statuses}
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Windows/Linux SSH scheduler for Experiment 7 GPU jobs")
    sub = parser.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory")
    inv.add_argument("--hosts", nargs="*", default=DEFAULT_HOSTS)
    inv.add_argument("--output", type=Path, required=True)
    inv.add_argument("--minimum-free-mib", type=int, default=12_000)
    inv.add_argument("--maximum-utilization", type=int, default=20)
    inv.add_argument("--ssh-timeout-seconds", type=int, default=60)

    plan = sub.add_parser("make-training-plan")
    plan.add_argument("--inventory", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--worktree", required=True)
    plan.add_argument("--commit", required=True)
    plan.add_argument("--python", required=True)
    plan.add_argument("--sources", required=True)
    plan.add_argument("--run-root", required=True)
    plan.add_argument("--stage", choices=("smoke", "pretrain", "finetune"), required=True)
    plan.add_argument("--pretrain-checkpoint")
    plan.add_argument("--seeds", type=int, nargs="*", default=[20260808, 20260809, 20260810])

    launch = sub.add_parser("launch")
    launch.add_argument("--plan", type=Path, required=True)
    launch.add_argument("--remote-python", required=True)
    launch.add_argument("--worker", required=True)

    status = sub.add_parser("status")
    status.add_argument("--plan", type=Path, required=True)
    status.add_argument("--remote-python", required=True)
    status.add_argument("--worker", required=True)

    args = parser.parse_args()
    if args.command == "inventory":
        inventory(
            args.hosts,
            args.output.resolve(),
            args.minimum_free_mib,
            args.maximum_utilization,
            args.ssh_timeout_seconds,
        )
    elif args.command == "make-training-plan":
        make_training_plan(
            args.inventory.resolve(),
            args.output.resolve(),
            PurePosixPath(args.worktree),
            args.commit,
            args.python,
            PurePosixPath(args.sources),
            PurePosixPath(args.run_root),
            args.stage,
            PurePosixPath(args.pretrain_checkpoint) if args.pretrain_checkpoint else None,
            args.seeds,
        )
    elif args.command == "launch":
        launch_plan(args.plan.resolve(), args.remote_python, PurePosixPath(args.worker))
    else:
        status_plan(args.plan.resolve(), args.remote_python, PurePosixPath(args.worker))


if __name__ == "__main__":
    main()
