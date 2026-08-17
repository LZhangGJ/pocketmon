from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_baseline_report(path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    epochs = report.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise ValueError(f"baseline report has no validation epochs: {path}")
    best_epoch = int(report.get("best", {}).get("epoch", epochs[-1]["epoch"]))
    row = next((item for item in epochs if int(item.get("epoch", -1)) == best_epoch), None)
    if not isinstance(row, dict) or not isinstance(row.get("validation"), dict):
        raise ValueError(
            f"baseline report lacks validation metrics for epoch {best_epoch}: {path}"
        )
    for metric in ("exactSemantic", "valueBrier"):
        if metric not in row["validation"]:
            raise ValueError(f"baseline report lacks {metric}: {path}")


def remote(host: str, command: str, *, timeout: int = 30) -> str:
    completed = subprocess.run(
        [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            f"lzhang@{host}",
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            shlex.quote(command),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(
            f"remote command failed host={host} rc={completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def gpus_idle(host: str, gpu_ids: tuple[int, int]) -> bool:
    query = remote(
        host,
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used "
        "--format=csv,noheader,nounits",
    )
    observed: dict[int, tuple[int, int]] = {}
    for line in query.splitlines():
        index, utilization, memory = (int(value.strip()) for value in line.split(","))
        observed[index] = (utilization, memory)
    return all(
        gpu in observed and observed[gpu][0] < 5 and observed[gpu][1] < 512
        for gpu in gpu_ids
    )


def launch_profile(
    *,
    host: str,
    profile: str,
    train_gpu: int,
    validation_gpu: int,
    sources: Path,
    checkpoint: Path,
    baseline_report: Path,
    output_root: Path,
    window_end: str,
) -> dict[str, Any]:
    control = output_root / "control"
    pidfile = control / f"{profile}.pid"
    logfile = control / f"{profile}.log"
    final_state = output_root / window_end / profile / "state.json"
    command = [
        "/homes/lzhang/mypath/new/envs/trans/bin/python",
        "-s",
        "/homes/lzhang/run_incremental_a100_bc_profile_20260813.py",
        "--profile",
        profile,
        "--gpu",
        str(train_gpu),
        "--validation-gpu",
        str(validation_gpu),
        "--sources",
        str(sources),
        "--initialize-from",
        str(checkpoint),
        "--baseline-report",
        str(baseline_report),
        "--output-root",
        str(output_root),
        "--window-end",
        window_end,
    ]
    shell_command = (
        f"mkdir -p {shlex.quote(str(control))}; "
        f"if test -s {shlex.quote(str(pidfile))} && "
        f"kill -0 $(cat {shlex.quote(str(pidfile))}) 2>/dev/null; then "
        f"echo ALREADY_RUNNING=$(cat {shlex.quote(str(pidfile))}); "
        f"elif test -s {shlex.quote(str(final_state))} && "
        f"grep -q '\"status\": \"complete' {shlex.quote(str(final_state))}; then "
        "echo ALREADY_COMPLETE; "
        "else "
        f"nohup {shlex.join(command)} >{shlex.quote(str(logfile))} 2>&1 </dev/null & "
        f"pid=$!; printf '%s\\n' \"$pid\" >{shlex.quote(str(pidfile))}; "
        "sleep 2; kill -0 \"$pid\"; echo STARTED=$pid; "
        "fi"
    )
    output = remote(host, shell_command)
    return {
        "profile": profile,
        "host": host,
        "trainGpu": train_gpu,
        "validationGpu": validation_gpu,
        "checkpoint": str(checkpoint),
        "checkpointSha256": sha256(checkpoint),
        "sources": str(sources),
        "command": command,
        "launcherOutput": output,
        "launchedAt": now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--daily-root", type=Path, required=True)
    parser.add_argument("--window-end", required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    main_root = args.main_root.resolve()
    daily_root = args.daily_root.resolve()
    output_root = Path(
        "/dataT0/Free/lzhang/pocketmon-runs/"
        "experiment7-universal-incremental-from-ppo-20260814"
    )
    state_path = main_root / "control/bc-from-ppo-0813-state.json"
    success = daily_root / "workers" / args.window_end / "SUCCESS"
    sources = Path(
        "/suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/"
        f"windows/{args.window_end}/tensordict-sources.json"
    )
    previous = Path(
        "/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/"
        "0812-d14-ram-npz-fast-20260813"
    )
    profiles = [
        {
            "profile": "large_256x6",
            "host": "10.113.13.71",
            "train_gpu": 0,
            "validation_gpu": 1,
            "checkpoint": main_root / "learners/universal_ppo_large_256x6/generation-000009/checkpoint.pt",
            # The validator needs per-epoch semantic/Brier metrics.  Those are
            # published by the validation worker, not the training report.
            "baseline_report": previous / "large_256x6/async-validation-report.json",
        },
        {
            "profile": "standard_1m",
            # Keep doraemon14 free for the active A08/Lucario learners.  The
            # two RTX 3090s on doraemon04 are otherwise idle and can train and
            # validate the standard profile without pre-empting PPO.
            "host": "10.113.13.54",
            "train_gpu": 0,
            "validation_gpu": 1,
            "checkpoint": main_root / "learners/universal_ppo_standard_1m/generation-000011/checkpoint.pt",
            "baseline_report": previous / "standard_1m/async-validation-report.json",
        },
    ]
    serializable_profiles = [
        {
            **row,
            "checkpoint": str(row["checkpoint"]),
            "baseline_report": str(row["baseline_report"]),
        }
        for row in profiles
    ]
    atomic_write(
        state_path,
        {
            "schemaVersion": 1,
            "status": "waiting_for_immutable_window",
            "windowEnd": args.window_end,
            "successMarker": str(success),
            "sources": str(sources),
            "profiles": serializable_profiles,
            "watcherPid": os.getpid(),
            "startedAt": now(),
        },
    )
    idle_receipts: dict[str, int] = {row["profile"]: 0 for row in profiles}
    while True:
        if success.is_file() and sources.is_file():
            for row in profiles:
                for required in (row["checkpoint"], row["baseline_report"]):
                    if not Path(required).is_file():
                        raise FileNotFoundError(required)
                validate_baseline_report(Path(row["baseline_report"]))
            all_idle = True
            for row in profiles:
                pair = (int(row["train_gpu"]), int(row["validation_gpu"]))
                if gpus_idle(str(row["host"]), pair):
                    idle_receipts[row["profile"]] += 1
                else:
                    idle_receipts[row["profile"]] = 0
                    all_idle = False
            if all_idle and all(value >= 2 for value in idle_receipts.values()):
                launches = [
                    launch_profile(
                        host=str(row["host"]),
                        profile=str(row["profile"]),
                        train_gpu=int(row["train_gpu"]),
                        validation_gpu=int(row["validation_gpu"]),
                        sources=sources,
                        checkpoint=Path(row["checkpoint"]),
                        baseline_report=Path(row["baseline_report"]),
                        output_root=output_root,
                        window_end=args.window_end,
                    )
                    for row in profiles
                ]
                atomic_write(
                    state_path,
                    {
                        "schemaVersion": 1,
                        "status": "incremental_bc_launched",
                        "windowEnd": args.window_end,
                        "sources": str(sources),
                        "launches": launches,
                        "launchedAt": now(),
                    },
                )
                return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
