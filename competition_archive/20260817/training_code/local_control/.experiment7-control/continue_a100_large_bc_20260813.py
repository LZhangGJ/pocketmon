#!/usr/bin/env python3
"""Continue one Universal BC capacity candidate on doraemon20 A100."""

from __future__ import annotations

import fcntl
import argparse
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
SOURCES = Path(
    "/dev/shm/lzhang-bc-capacity-a100-20260812/universal-10d-sources-ram.json"
)
PYTHON = Path("/homes/lzhang/mypath/new/envs/trans/bin/python")
TRAINER = Path(
    "/homes/lzhang/worktrees/experiment7-async-4c45f89/"
    "experiment7/integration/train_universal_bc.py"
)
START_EPOCH = 5
MAX_EPOCH = 12
MIN_SEMANTIC_DELTA = 0.002
MAX_BRIER_INCREASE = 0.005
PATIENCE = 2


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def metrics(report_path: Path) -> tuple[float, float]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validation = report["epochs"][-1]["validation"]
    return float(validation["exactSemantic"]), float(validation["valueBrier"])


def learning_rate(epoch: int) -> float:
    if epoch <= 6:
        return 1e-4
    if epoch <= 8:
        return 5e-5
    return 2.5e-5


PROFILES = {
    "standard_1m": {
        "d_model": 128,
        "heads": 4,
        "layers": 3,
        "ff_dim": 384,
    },
    "large_256x6": {
        "d_model": 256,
        "heads": 8,
        "layers": 6,
        "ff_dim": 1024,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--max-epoch", type=int, default=MAX_EPOCH)
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    baseline = BASE / args.profile
    root = BASE / f"{args.profile}-continuation"
    gpu = args.gpu
    root.mkdir(parents=True, exist_ok=True)
    lock_handle = (root / "controller.lock").open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("CONTROLLER_ALREADY_RUNNING", flush=True)
        return 0

    for required in (SOURCES, PYTHON, TRAINER, baseline / "best_model.pt"):
        if not required.exists():
            raise FileNotFoundError(required)

    source_manifest = json.loads(SOURCES.read_text(encoding="utf-8"))
    missing = []
    for row in source_manifest.get("datasets", []):
        for key in ("features", "tokenCache", "sequenceCache", "identityCache"):
            value = row.get(key)
            if value and not Path(value).exists():
                missing.append(value)
    if missing:
        raise RuntimeError(f"RAM cache incomplete: {missing[:3]}")

    baseline_report = baseline / "training_report.json"
    previous_score, previous_brier = metrics(baseline_report)
    previous_checkpoint = baseline / "best_model.pt"
    best_score = previous_score
    best_brier = previous_brier
    best_checkpoint = previous_checkpoint
    best_epoch = 4
    stagnant = 0
    brier_regressions = 0
    history: list[dict] = []

    state_path = root / "continuation_state.json"
    for epoch in range(START_EPOCH, args.max_epoch + 1):
        epoch_root = root / f"epoch-{epoch:02d}"
        report_path = epoch_root / "training_report.json"
        checkpoint = epoch_root / "best_model.pt"
        lr = learning_rate(epoch)
        if not report_path.is_file() or not checkpoint.is_file():
            epoch_root.mkdir(parents=True, exist_ok=True)
            command = [
                str(PYTHON),
                "-s",
                str(TRAINER),
                "--sources",
                str(SOURCES),
                "--output-dir",
                str(epoch_root),
                "--initialize-from",
                str(previous_checkpoint),
                "--device",
                "cuda:0",
                "--seed",
                str(20260812 + epoch),
                "--epochs",
                "1",
                "--batch-size",
                "256",
                "--learning-rate",
                str(lr),
                "--weight-decay",
                "1e-4",
                "--value-loss-weight",
                "0.05",
                "--prefetch-batches",
                "6",
                "--prefetch-workers",
                "2",
                "--d-model",
                str(profile["d_model"]),
                "--heads",
                str(profile["heads"]),
                "--layers",
                str(profile["layers"]),
                "--ff-dim",
                str(profile["ff_dim"]),
                "--dropout",
                "0.05",
            ]
            env = dict(os.environ)
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "PYTHONNOUSERSITE": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                }
            )
            with (epoch_root / "train.log").open("a", encoding="utf-8") as log:
                completed = subprocess.run(
                    ["ionice", "-c2", "-n7", "nice", "-n", "10", *command],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=False,
                )
            if completed.returncode:
                write_json(
                    root / "FAILED.json",
                    {"at": now(), "epoch": epoch, "returnCode": completed.returncode},
                )
                return completed.returncode

        score, brier = metrics(report_path)
        delta = score - previous_score
        brier_delta = brier - previous_brier
        stagnant = stagnant + 1 if delta < MIN_SEMANTIC_DELTA else 0
        brier_regressions = (
            brier_regressions + 1 if brier_delta > MAX_BRIER_INCREASE else 0
        )
        if score > best_score:
            best_score = score
            best_brier = brier
            best_checkpoint = checkpoint
            best_epoch = epoch
        history.append(
            {
                "absoluteEpoch": epoch,
                "learningRate": lr,
                "exactSemantic": score,
                "delta": delta,
                "valueBrier": brier,
                "brierDelta": brier_delta,
                "checkpoint": str(checkpoint),
            }
        )
        state = {
            "schemaVersion": 1,
            "updatedAt": now(),
            "status": "training",
            "profile": args.profile,
            "gpu": gpu,
            "baselineEpoch": 4,
            "baselineScore": metrics(baseline_report)[0],
            "bestEpoch": best_epoch,
            "bestScore": best_score,
            "bestBrier": best_brier,
            "bestCheckpoint": str(best_checkpoint),
            "stagnantEpochs": stagnant,
            "brierRegressionEpochs": brier_regressions,
            "history": history,
        }
        write_json(state_path, state)
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
        previous_score, previous_brier = score, brier
        previous_checkpoint = checkpoint
        if stagnant >= PATIENCE or brier_regressions >= PATIENCE:
            state["status"] = "early_stopped"
            state["stopReason"] = (
                "semantic_plateau" if stagnant >= PATIENCE else "brier_regression"
            )
            write_json(state_path, state)
            break
    else:
        state["status"] = "max_epoch_reached"
        write_json(state_path, state)

    selected = root / "selected_best_model.pt"
    temporary = selected.with_suffix(".pt.tmp")
    shutil.copyfile(best_checkpoint, temporary)
    os.replace(temporary, selected)
    state["selectedCheckpoint"] = str(selected)
    state["completedAt"] = now()
    write_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
