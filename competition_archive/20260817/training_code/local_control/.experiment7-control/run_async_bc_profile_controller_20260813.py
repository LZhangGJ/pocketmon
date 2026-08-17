#!/usr/bin/env python3
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


RUNTIME = Path("/tmp/experiment7-async-bc-runtime-20260813")
DEFAULT_SOURCES = Path("/dev/shm/lzhang-bc-capacity-a100-20260812/universal-10d-sources-ram.json")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--train-gpu", required=True)
    parser.add_argument("--validation-gpu", required=True)
    parser.add_argument("--start-epoch", type=int, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    args = parser.parse_args()
    # Keep a stable procfs handle to this controller's already-loaded Python
    # executable.  /proc/self/exe would resolve to ionice after the wrapper
    # starts, so it cannot be used as the child executable path.
    python = Path(f"/proc/{os.getpid()}/exe")

    previous = args.previous_root.resolve()
    output = args.output_root.resolve() / args.profile
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "controller.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ASYNC_PROFILE_ALREADY_RUNNING", flush=True)
        return 0
    baseline_checkpoint = previous / "best_model.pt"
    baseline_report = previous / "training_report.json"
    sources = args.sources.resolve()
    # Do not stat /proc/self/exe: resolving it reaches the interpreter's
    # original shared-filesystem path and defeats the local-runtime escape.
    for required in (sources, baseline_checkpoint, baseline_report, RUNTIME / "train_universal_bc_async.py", RUNTIME / "validate_universal_bc_async.py"):
        if not required.exists():
            raise FileNotFoundError(required)
    state_path = output / "controller-state.json"
    write(state_path, {"status": "launching", "profile": args.profile, "startedAt": now()})

    common_env = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    train_env = dict(common_env, CUDA_VISIBLE_DEVICES=args.train_gpu)
    validation_env = dict(common_env, CUDA_VISIBLE_DEVICES=args.validation_gpu)
    trainer = [
        "ionice", "-c2", "-n7", "nice", "-n", "10", str(python), "-s",
        str(RUNTIME / "train_universal_bc_async.py"),
        "--sources", str(sources), "--output-dir", str(output),
        "--initialize-from", str(baseline_checkpoint), "--device", "cuda:0",
        "--epoch-start", str(args.start_epoch), "--epochs", "1000000",
        "--batch-size", "256", "--learning-rate", "0.00005",
        "--learning-rate-schedule", "9=0.000025", "--weight-decay", "1e-4",
        "--value-loss-weight", "0.05", "--prefetch-batches", "6",
        "--prefetch-workers", "2", "--max-pending-validations", "2",
    ]
    validator = [
        "ionice", "-c2", "-n7", "nice", "-n", "10", str(python), "-s",
        str(RUNTIME / "validate_universal_bc_async.py"),
        "--sources", str(sources), "--output-dir", str(output),
        "--baseline-report", str(baseline_report),
        "--baseline-checkpoint", str(baseline_checkpoint), "--device", "cuda:0",
        "--batch-size", "256", "--patience", "3",
        "--min-semantic-delta", "0.002", "--max-brier-increase", "0.005",
    ]
    with (output / "train.log").open("a") as train_log, (output / "validation.log").open("a") as validation_log:
        train_process = subprocess.Popen(trainer, stdout=train_log, stderr=subprocess.STDOUT, env=train_env)
        write(state_path, {
            "status": "training_waiting_for_first_checkpoint", "profile": args.profile,
            "trainPid": train_process.pid, "trainGpu": args.train_gpu,
            "validationGpu": args.validation_gpu, "startedAt": now(),
        })
        # Avoid importing PyTorch and mapping the validation set while the
        # trainer is still doing its cold start.  The validator has nothing to
        # consume before the first checkpoint, and concurrent cold starts are
        # especially harmful on shared-filesystem hosts.
        checkpoints = output / "checkpoints"
        while train_process.poll() is None and not any(checkpoints.glob("epoch_*.pt")):
            time.sleep(5)
        if train_process.poll() is not None:
            train_code = train_process.returncode
            write(state_path, {"status": "failed_before_first_checkpoint", "trainReturnCode": train_code, "failedAt": now()})
            return train_code or 1
        validation_process = subprocess.Popen(validator, stdout=validation_log, stderr=subprocess.STDOUT, env=validation_env)
        write(state_path, {
            "status": "training_and_validating", "profile": args.profile,
            "trainPid": train_process.pid, "validationPid": validation_process.pid,
            "trainGpu": args.train_gpu, "validationGpu": args.validation_gpu,
            "startedAt": now(),
        })
        train_code = train_process.wait()
        validation_code = validation_process.wait()
    if train_code or validation_code:
        write(state_path, {"status": "failed", "trainReturnCode": train_code, "validationReturnCode": validation_code, "failedAt": now()})
        return train_code or validation_code
    selected = output / "best_model.pt"
    if not selected.is_file():
        raise FileNotFoundError(selected)
    destination = previous / "selected_best_model.pt"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copyfile(selected, temporary)
    os.replace(temporary, destination)
    final = {
        "schemaVersion": 3, "status": "complete",
        "mode": "persistent_training_with_async_validation",
        "profile": args.profile, "selectedCheckpoint": str(destination),
        "asyncRoot": str(output), "completedAt": now(),
    }
    write(previous / "continuation_state.json", final)
    write(state_path, final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
