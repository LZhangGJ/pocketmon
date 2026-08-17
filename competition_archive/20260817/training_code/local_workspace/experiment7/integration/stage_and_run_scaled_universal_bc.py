from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def copy_tree(source: Path, target: Path) -> str:
    if not source.is_dir():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--delete-delay", f"{source}/", f"{target}/"], check=True)
    return str(target)


def write_state(final: Path, attempt: Path, payload: dict) -> None:
    atomic_json(attempt / "state.json", payload)
    atomic_json(final / "state.json", payload)
    atomic_json(final / "latest.json", payload)


def reusable_local_sources(stage_root: Path) -> Path | None:
    manifest = stage_root / "tensordict-sources.local.json"
    if not (stage_root / "SUCCESS").is_file() or not manifest.is_file():
        return None
    payload = json.loads(manifest.read_text())
    required = [Path(payload["referenceRoot"]), Path(payload["engineCatalog"]["path"])]
    for dataset in payload["datasets"]:
        required.extend(
            Path(dataset[key])
            for key in ("features", "tokenCache", "sequenceCache", "identityCache")
        )
    return manifest if all(path.exists() for path in required) else None


def stage_sources(
    source_manifest: Path,
    stage_root: Path,
    workers: int,
    final: Path,
    attempt: Path,
) -> Path:
    reusable = reusable_local_sources(stage_root)
    if reusable is not None:
        write_state(final, attempt, {
            "status": "staged_reused",
            "completedAt": now(),
            "sources": str(reusable),
        })
        return reusable
    payload = json.loads(source_manifest.read_text())
    jobs = []
    for dataset in payload["datasets"]:
        for key in ("features", "tokenCache", "sequenceCache", "identityCache"):
            source = Path(dataset[key])
            target = stage_root / "datasets" / dataset["name"] / key
            jobs.append((dataset, key, source, target))
    write_state(final, attempt, {
        "status": "staging",
        "startedAt": now(),
        "copyJobs": len(jobs),
    })
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [(dataset, key, target, executor.submit(copy_tree, source, target)) for dataset, key, source, target in jobs]
        for dataset, key, target, future in futures:
            future.result()
            dataset[key] = str(target)

    reference_target = stage_root / "reference"
    copy_tree(Path(payload["referenceRoot"]), reference_target)
    payload["referenceRoot"] = str(reference_target)
    engine_source = Path(payload["engineCatalog"]["path"])
    engine_target = stage_root / "engine_catalog.json"
    shutil.copyfile(engine_source, engine_target)
    payload["engineCatalog"]["path"] = str(engine_target)
    local_manifest = stage_root / "tensordict-sources.local.json"
    atomic_json(local_manifest, payload)
    (stage_root / "SUCCESS").write_text(now() + "\n")
    write_state(final, attempt, {
        "status": "staged",
        "completedAt": now(),
        "sources": str(local_manifest),
    })
    return local_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/homes/lzhang/mypath/new/envs/trans/bin/python"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--train-gpu", default="0")
    parser.add_argument("--validation-gpu", default="1")
    parser.add_argument("--copy-workers", type=int, default=4)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--expected-card-vocab", type=int, required=True)
    parser.add_argument("--lock-name", default="controller.lock")
    args = parser.parse_args()

    final = args.output_root.resolve() / "ultra_512x8_31m"
    final.mkdir(parents=True, exist_ok=True)
    attempt = final / "attempts" / args.attempt_id
    attempt.mkdir(parents=True, exist_ok=True)
    if Path(args.lock_name).name != args.lock_name:
        raise ValueError("lock-name must be a basename")
    lock = (final / args.lock_name).open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("SCALED_PROFILE_ALREADY_RUNNING", flush=True)
        return 0
    plan = {
        "schemaVersion": 1,
        "profile": "ultra_512x8_31m",
        "attemptId": args.attempt_id,
        "host": socket.gethostname(),
        "controllerPid": os.getpid(),
        "lockName": args.lock_name,
        "createdAt": now(),
        "trainGpu": args.train_gpu,
        "validationGpu": args.validation_gpu,
        "expectedCardVocab": args.expected_card_vocab,
        "targetParameterCount": 30_724_612,
        "modelConfig": {
            "dModel": 512,
            "heads": 8,
            "layers": 8,
            "ffDim": 2048,
        },
        "stageRoot": str(args.stage_root.resolve()),
        "scratchRoot": str(
            args.scratch_root.resolve()
            if args.scratch_root is not None
            else args.stage_root.resolve() / "attempts"
        ),
        "runtime": str(args.runtime.resolve()),
        "python": str(args.python.resolve()),
        "outputRoot": str(final),
        "initializeFrom": (
            str(args.initialize_from.resolve())
            if args.initialize_from is not None
            else None
        ),
    }
    atomic_json(attempt / "plan.json", plan)
    atomic_json(final / "plan.json", plan)
    local_sources = stage_sources(
        args.sources.resolve(), args.stage_root.resolve(), args.copy_workers,
        final, attempt,
    )
    sources_payload = json.loads(local_sources.read_text())
    actual_card_vocab = int(sources_payload["engineCatalog"]["cardVocab"])
    if actual_card_vocab != args.expected_card_vocab:
        write_state(final, attempt, {
            "status": "failed_preflight",
            "attemptId": args.attempt_id,
            "expectedCardVocab": args.expected_card_vocab,
            "actualCardVocab": actual_card_vocab,
            "failedAt": now(),
        })
        return 2
    scratch_base = (
        args.scratch_root.resolve()
        if args.scratch_root is not None
        else args.stage_root.resolve() / "attempts"
    )
    scratch = scratch_base / args.attempt_id / "training"
    scratch.mkdir(parents=True, exist_ok=True)
    python = args.python.resolve()
    if not python.is_file():
        raise FileNotFoundError(python)
    trainer_cmd = [
        "ionice", "-c2", "-n7", "nice", "-n", "10", str(python), "-s",
        str(args.runtime / "train_universal_bc_async_scaled.py"),
        "--sources", str(local_sources), "--output-dir", str(scratch), "--device", "cuda:0",
        "--batch-size", "128", "--learning-rate", "0.0001",
        "--learning-rate-schedule", "3=0.00005", "--learning-rate-schedule", "6=0.000025",
        "--d-model", "512", "--heads", "8", "--layers", "8", "--ff-dim", "2048",
        "--expected-card-vocab", str(args.expected_card_vocab),
    ]
    if args.initialize_from is not None:
        initialization = args.initialize_from.resolve()
        if not initialization.is_file():
            raise FileNotFoundError(initialization)
        trainer_cmd.extend(["--initialize-from", str(initialization)])
    validator_cmd = [
        "ionice", "-c2", "-n7", "nice", "-n", "10", str(python), "-s",
        str(args.runtime / "validate_universal_bc_async_scaled.py"),
        "--sources", str(local_sources), "--output-dir", str(scratch),
        "--baseline-report", str(args.baseline_report.resolve()), "--device", "cuda:0",
        "--batch-size", "128", "--minimum-epochs", "6", "--patience", "3",
    ]
    base_env = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (attempt / "train.log").open("a") as train_log, (attempt / "validation.log").open("a") as validation_log:
        trainer = subprocess.Popen(trainer_cmd, stdout=train_log, stderr=subprocess.STDOUT, env=dict(base_env, CUDA_VISIBLE_DEVICES=args.train_gpu))
        validator = subprocess.Popen(validator_cmd, stdout=validation_log, stderr=subprocess.STDOUT, env=dict(base_env, CUDA_VISIBLE_DEVICES=args.validation_gpu))
        write_state(final, attempt, {
            "status": "training_and_validating", "startedAt": now(), "profile": "ultra_512x8_31m",
            "attemptId": args.attempt_id, "host": socket.gethostname(),
            "controllerPid": os.getpid(),
            "trainPid": trainer.pid, "validationPid": validator.pid,
            "trainGpu": args.train_gpu, "validationGpu": args.validation_gpu,
            "sources": str(local_sources), "scratch": str(scratch),
            "cardVocab": actual_card_vocab,
            "targetParameterCount": 30_724_612,
            "initializeFrom": (
                str(args.initialize_from.resolve())
                if args.initialize_from is not None
                else None
            ),
        })
        while trainer.poll() is None or validator.poll() is None:
            if trainer.poll() not in (None, 0) and validator.poll() is None:
                validator.terminate()
            if validator.poll() not in (None, 0) and trainer.poll() is None:
                trainer.terminate()
            time.sleep(2)
    if trainer.returncode or validator.returncode:
        write_state(final, attempt, {
            "status": "failed", "attemptId": args.attempt_id,
            "controllerPid": os.getpid(), "trainPid": trainer.pid,
            "validationPid": validator.pid,
            "trainReturnCode": trainer.returncode,
            "validationReturnCode": validator.returncode, "failedAt": now(),
        })
        return trainer.returncode or validator.returncode
    for name in ("best_model.pt", "async-training-report.json", "async-validation-report.json"):
        shutil.copyfile(scratch / name, final / name)
    write_state(final, attempt, {
        "status": "complete_waiting_for_parity_and_screening",
        "attemptId": args.attempt_id, "completedAt": now(),
        "checkpoint": str(final / "best_model.pt"),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
