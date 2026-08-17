#!/usr/bin/env python3
"""Continue one Universal BC candidate in one data-loading/training process."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE = Path(
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/"
    "capacity-comparison-a100-ram-prefetch-b256"
)
OUTPUT_BASE = Path(
    "/tmp/lzhang-bc-capacity-a100-20260813"
)
SOURCES = Path("/dev/shm/lzhang-bc-capacity-a100-20260812/universal-10d-sources-ram.json")
PYTHON = Path("/homes/lzhang/mypath/new/envs/trans/bin/python")
TRAINER = Path(
    "/homes/lzhang/worktrees/experiment7-async-4c45f89/"
    "experiment7/integration/train_universal_bc.py"
)
PROFILES = {
    "standard_1m": {"d_model": 128, "heads": 4, "layers": 3, "ff_dim": 384},
    "large_256x6": {"d_model": 256, "heads": 8, "layers": 6, "ff_dim": 1024},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def report_best(path: Path) -> tuple[int, float, float, Path]:
    report = json.loads(path.read_text())
    best_epoch = int(report["best"]["epoch"])
    row = next(row for row in report["epochs"] if int(row["epoch"]) == best_epoch)
    return (
        best_epoch,
        float(row["validation"]["exactSemantic"]),
        float(row["validation"]["valueBrier"]),
        Path(report["best"]["path"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--start-epoch", type=int, default=5)
    parser.add_argument("--max-epoch", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.max_epoch < args.start_epoch:
        parser.error("--max-epoch must be at least --start-epoch")

    profile = PROFILES[args.profile]
    baseline = BASE / args.profile
    root = OUTPUT_BASE / f"{args.profile}-persistent-continuation"
    root.mkdir(parents=True, exist_ok=True)
    lock_handle = (root / "controller.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("CONTROLLER_ALREADY_RUNNING", flush=True)
        return 0

    baseline_report = baseline / "training_report.json"
    baseline_checkpoint = baseline / "best_model.pt"
    for required in (SOURCES, PYTHON, TRAINER, baseline_report, baseline_checkpoint):
        if not required.exists():
            raise FileNotFoundError(required)

    state_path = root / "continuation_state.json"
    write_json(
        state_path,
        {
            "schemaVersion": 2,
            "status": "training",
            "mode": "single_persistent_process",
            "profile": args.profile,
            "gpu": args.gpu,
            "startEpoch": args.start_epoch,
            "maxEpoch": args.max_epoch,
            "startedAt": now(),
        },
    )

    command = [
        "ionice", "-c2", "-n7", "nice", "-n", "10",
        str(PYTHON), "-s", str(TRAINER),
        "--sources", str(SOURCES),
        "--output-dir", str(root),
        "--initialize-from", str(baseline_checkpoint),
        "--device", "cuda:0",
        "--seed", "20260817",
        "--epochs", str(args.max_epoch - args.start_epoch + 1),
        "--epoch-start", str(args.start_epoch),
        "--batch-size", "256",
        "--learning-rate", "0.0001",
        "--learning-rate-schedule", "5=0.0001",
        "--learning-rate-schedule", "7=0.00005",
        "--learning-rate-schedule", "9=0.000025",
        "--early-stop-patience", "3",
        "--early-stop-min-semantic-delta", "0.002",
        "--early-stop-max-brier-increase", "0.005",
        "--early-stop-baseline-report", str(baseline_report),
        "--weight-decay", "1e-4",
        "--value-loss-weight", "0.05",
        "--prefetch-batches", "6",
        "--prefetch-workers", "2",
        "--d-model", str(profile["d_model"]),
        "--heads", str(profile["heads"]),
        "--layers", str(profile["layers"]),
        "--ff-dim", str(profile["ff_dim"]),
        "--dropout", "0.05",
    ]
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    with (root / "train.log").open("a", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    if completed.returncode:
        write_json(state_path, {"schemaVersion": 2, "status": "failed", "returnCode": completed.returncode, "failedAt": now()})
        return completed.returncode

    baseline_best = report_best(baseline_report)
    continuation_best = report_best(root / "training_report.json")
    selected = continuation_best if continuation_best[1] > baseline_best[1] else baseline_best
    selected_path = root / "selected_best_model.pt"
    temporary = selected_path.with_suffix(".pt.tmp")
    shutil.copyfile(selected[3], temporary)
    os.replace(temporary, selected_path)
    state = {
        "schemaVersion": 2,
        "status": "complete",
        "mode": "single_persistent_process",
        "profile": args.profile,
        "gpu": args.gpu,
        "baseline": {"epoch": baseline_best[0], "exactSemantic": baseline_best[1], "valueBrier": baseline_best[2]},
        "continuationBest": {"epoch": continuation_best[0], "exactSemantic": continuation_best[1], "valueBrier": continuation_best[2]},
        "selectedEpoch": selected[0],
        "selectedCheckpoint": str(selected_path),
        "completedAt": now(),
    }
    write_json(state_path, state)
    print(json.dumps(state), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
