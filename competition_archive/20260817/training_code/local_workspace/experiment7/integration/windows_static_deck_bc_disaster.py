from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRICT_PREDICATE = "is_clean == 1 and float(min_score) > 1000.0"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def gpu_sample() -> tuple[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(int(value.strip()) for value in completed.stdout.strip().split(","))  # type: ignore[return-value]


def gpu_idle(sample: tuple[int, int]) -> bool:
    # WDDM reserves about 2.4 GiB even with no compute process.
    return sample[0] <= 3072 and sample[1] <= 20


def build_sources(local_root: Path, profile: str, reference_root: Path) -> dict[str, Any] | None:
    day_root = local_root / "day-shards"
    datasets = []
    if not day_root.is_dir():
        return None
    for day in sorted(path for path in day_root.iterdir() if path.is_dir()):
        shard = day / profile
        receipt_path = shard / "specialist-receipt.json"
        if not (shard / "SUCCESS").is_file() or not receipt_path.is_file():
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("strictPredicate") != STRICT_PREDICATE:
            raise ValueError(f"non-strict specialist shard: {receipt_path}")
        if float(receipt.get("scoreMin", 1000.0)) <= 1000.0:
            raise ValueError(f"score boundary violation: {receipt_path}")
        datasets.append(
            {
                "name": day.name,
                "root": str(shard),
                "features": str(shard / "features_tensordict"),
                "decisions": str(shard / "decisions.jsonl.gz"),
                "tokenCache": str(shard / "token_cache"),
                "sequenceCache": str(shard / "sequence_cache"),
                "identityCache": str(shard / "identity_cache"),
                "specialistReceipt": str(receipt_path),
                "summary": {
                    "sourceEpisodes": int(receipt["episodes"]),
                    "decisions": int(receipt["decisions"]),
                },
            }
        )
    if len(datasets) != 10:
        return None
    return {
        "schemaVersion": 1,
        "kind": "experiment7_universal_bc",
        "staticProfile": f"10d-deck-specialist-static-bc:{profile}",
        "referenceRoot": str(reference_root),
        "strictPredicate": STRICT_PREDICATE,
        "minGameScoreExclusive": 1000.0,
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    local_root = args.local_root.resolve()
    persistent = args.persistent_root.resolve()
    state = args.state.resolve()
    integration = persistent / "runtime" / "source" / "experiment7" / "integration"
    reference = persistent / "runtime" / "source" / "experiment7" / "reference"
    config = persistent / "runtime" / "source" / "experiment7" / "config" / "static_deck_bc_10d_20260815.json"
    initializer = persistent / "checkpoints" / "universal-large-g51.pt"
    python = persistent / "runtime" / "venv" / "Scripts" / "python.exe"
    training = persistent / "profiles" / args.profile / "disaster-training"
    logs = persistent / "profiles" / args.profile / "disaster-logs"
    sources_path = local_root / "tensordict-sources.windows.json"
    training.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    atomic_json(
        state,
        {
            "schemaVersion": 1,
            "status": "waiting_immutable_ten_day_sources",
            "profile": args.profile,
            "localRoot": str(local_root),
            "pid": os.getpid(),
            "observedAt": now(),
        },
    )
    while True:
        sources = build_sources(local_root, args.profile, reference)
        if sources is not None:
            break
        time.sleep(args.poll_seconds)
    atomic_json(sources_path, sources)

    samples = [gpu_sample()]
    time.sleep(10)
    samples.append(gpu_sample())
    if not all(gpu_idle(sample) for sample in samples):
        raise RuntimeError(f"Windows GPU0 is not idle after two samples: {samples}")

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(integration),
            "PIP_CACHE_DIR": str(local_root / "cache" / "pip"),
            "TMP": str(local_root / "tmp"),
            "TEMP": str(local_root / "tmp"),
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )
    (local_root / "tmp").mkdir(parents=True, exist_ok=True)
    validator_log = (logs / "validator.log").open("a", encoding="utf-8")
    validator = subprocess.Popen(
        [
            str(python),
            str(integration / "validate_static_deck_bc_async.py"),
            "--config",
            str(config),
            "--sources",
            str(sources_path),
            "--output-dir",
            str(training),
            "--initialize-from",
            str(initializer),
            "--device",
            "cuda:0",
            "--batch-size",
            str(args.batch_size),
        ],
        stdout=validator_log,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    while not (training / "async-validation-report.json").is_file():
        if validator.poll() is not None:
            raise RuntimeError(f"validator exited before baseline: {validator.returncode}")
        time.sleep(5)
    trainer_log = (logs / "trainer.log").open("a", encoding="utf-8")
    trainer = subprocess.Popen(
        [
            str(python),
            str(integration / "train_static_deck_bc_async.py"),
            "--config",
            str(config),
            "--sources",
            str(sources_path),
            "--output-dir",
            str(training),
            "--initialize-from",
            str(initializer),
            "--device",
            "cuda:0",
            "--batch-size",
            str(args.batch_size),
            "--prefetch-batches",
            "0",
            "--prefetch-workers",
            "1",
        ],
        stdout=trainer_log,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    atomic_json(
        state,
        {
            "schemaVersion": 1,
            "status": "training_with_async_validation",
            "profile": args.profile,
            "controllerPid": os.getpid(),
            "trainerPid": trainer.pid,
            "validatorPid": validator.pid,
            "gpu": 0,
            "batchSize": args.batch_size,
            "sources": str(sources_path),
            "formalTrainingStarted": True,
            "observedAt": now(),
        },
    )
    trainer_rc = trainer.wait()
    validator_rc = validator.wait()
    validator_log.close()
    trainer_log.close()
    best = training / "best_model.pt"
    if trainer_rc or validator_rc or not best.is_file():
        raise SystemExit(1)
    frozen = persistent / "frozen" / args.profile / "best_model.pt"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    temporary = frozen.with_name(f".{frozen.name}.{os.getpid()}.tmp")
    shutil.copy2(best, temporary)
    os.replace(temporary, frozen)
    atomic_json(
        state,
        {
            "schemaVersion": 1,
            "status": "training_complete",
            "profile": args.profile,
            "bestModel": str(best),
            "frozenCopy": str(frozen),
            "staticFrozen": True,
            "ppoUpdatesAllowed": False,
            "completedAt": now(),
        },
    )


if __name__ == "__main__":
    main()
