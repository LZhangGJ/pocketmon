#!/usr/bin/env python3
"""Run one TensorDict-window profile with persistent async validation."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PYTHON = Path("/homes/lzhang/mypath/new/envs/trans/bin/python")
RUNTIME = Path("/homes/lzhang/experiment7-async-bc-runtime")
TRAINER = RUNTIME / "train_universal_bc_async.py"
VALIDATOR = RUNTIME / "validate_universal_bc_async.py"
PROFILES = {
    "standard_1m": {"d_model": 128, "heads": 4, "layers": 3, "ff_dim": 384},
    "large_256x6": {"d_model": 256, "heads": 8, "layers": 6, "ff_dim": 1024},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--validation-gpu", required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window-end", required=True)
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    final_root = args.output_root.resolve() / args.window_end / args.profile
    scratch = Path("/tmp/lzhang-bc-tensordict-incremental") / args.window_end / args.profile
    final_root.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    lock_handle = (final_root / "controller.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("INCREMENTAL_PROFILE_ALREADY_RUNNING", flush=True)
        return 0

    for required in (PYTHON, TRAINER, VALIDATOR, args.sources, args.initialize_from, args.baseline_report):
        if not required.exists():
            raise FileNotFoundError(required)
    state_path = final_root / "state.json"
    write(
        state_path,
        {
            "schemaVersion": 1,
            "status": "training",
            "mode": "persistent_training_with_async_checkpoint_validation",
            "storage": "TensorDict-style read-only memmap",
            "profile": args.profile,
            "gpu": args.gpu,
            "validationGpu": args.validation_gpu,
            "windowEnd": args.window_end,
            "sources": str(args.sources.resolve()),
            "initializeFrom": str(args.initialize_from.resolve()),
            "startedAt": now(),
        },
    )
    command = [
        "ionice", "-c2", "-n7", "nice", "-n", "10",
        str(PYTHON), "-s", str(TRAINER),
        "--sources", str(args.sources.resolve()),
        "--output-dir", str(scratch),
        "--initialize-from", str(args.initialize_from.resolve()),
        "--device", "cuda:0",
        "--seed", "20260818", "--epoch-start", "1",
        "--epochs", "1000000",
        "--batch-size", "256",
        "--learning-rate", "0.00005",
        "--learning-rate-schedule", "1=0.00005",
        "--learning-rate-schedule", "3=0.000025",
        "--learning-rate-schedule", "5=0.0000125",
        "--weight-decay", "1e-4",
        "--value-loss-weight", "0.05",
        "--prefetch-batches", "6",
        "--prefetch-workers", "2",
        "--max-pending-validations", "2",
    ]
    validation_command = [
        "ionice", "-c2", "-n7", "nice", "-n", "10",
        str(PYTHON), "-s", str(VALIDATOR),
        "--sources", str(args.sources.resolve()),
        "--output-dir", str(scratch),
        "--baseline-report", str(args.baseline_report.resolve()),
        "--baseline-checkpoint", str(args.initialize_from.resolve()),
        "--device", "cuda:0", "--batch-size", "256", "--patience", "3",
        "--min-semantic-delta", "0.001", "--max-brier-increase", "0.005",
    ]
    train_env = dict(os.environ)
    train_env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    validation_env = dict(train_env, CUDA_VISIBLE_DEVICES=args.validation_gpu)
    with (final_root / "train.log").open("a") as train_log, (final_root / "validation.log").open("a") as validation_log:
        trainer = subprocess.Popen(command, stdout=train_log, stderr=subprocess.STDOUT, env=train_env)
        validator = subprocess.Popen(validation_command, stdout=validation_log, stderr=subprocess.STDOUT, env=validation_env)
        write(state_path, {
            "schemaVersion": 1, "status": "training_and_validating", "profile": args.profile,
            "trainPid": trainer.pid, "validationPid": validator.pid,
            "trainGpu": args.gpu, "validationGpu": args.validation_gpu,
            "sources": str(args.sources.resolve()), "startedAt": now(),
        })
        while trainer.poll() is None or validator.poll() is None:
            if trainer.poll() not in (None, 0) and validator.poll() is None:
                validator.terminate()
            if validator.poll() not in (None, 0) and trainer.poll() is None:
                trainer.terminate()
            time.sleep(2)
        train_code = trainer.returncode
        validation_code = validator.returncode
    if train_code or validation_code:
        write(state_path, {
            "schemaVersion": 1, "status": "failed", "trainReturnCode": train_code,
            "validationReturnCode": validation_code, "failedAt": now(),
        })
        return train_code or validation_code

    for name in ("best_model.pt", "training_report.json"):
        source = scratch / name
        if not source.is_file():
            raise FileNotFoundError(source)
        temporary = final_root / f".{name}.tmp"
        shutil.copyfile(source, temporary)
        os.replace(temporary, final_root / name)
    write(
        state_path,
        {
            "schemaVersion": 1,
            "status": "complete_waiting_for_parity_and_screening",
            "mode": "persistent_training_with_async_checkpoint_validation",
            "profile": args.profile,
            "windowEnd": args.window_end,
            "checkpoint": str(final_root / "best_model.pt"),
            "report": str(final_root / "training_report.json"),
            "completedAt": now(),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
