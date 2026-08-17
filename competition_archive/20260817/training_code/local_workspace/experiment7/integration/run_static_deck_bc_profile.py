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


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def acquire(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "createdAt": now()}, handle)
        handle.write("\n")


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed rc={completed.returncode}: {command}")


def specialist_sources(
    manifest: dict[str, Any],
    archetype: str,
    local_root: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    datasets = []
    for row in manifest["datasets"]:
        shard = local_root / "day-shards" / str(row["name"]) / archetype
        receipt = load_json(shard / "specialist-receipt.json")
        datasets.append({
            "name": str(row["name"]),
            "root": str(shard),
            "features": str(shard / "features_tensordict"),
            "decisions": str(shard / "decisions.jsonl.gz"),
            "tokenCache": str(shard / "token_cache"),
            "sequenceCache": str(shard / "sequence_cache"),
            "identityCache": str(shard / "identity_cache"),
            "specialistReceipt": str(shard / "specialist-receipt.json"),
            "summary": {"sourceEpisodes": receipt["episodes"], "decisions": receipt["decisions"]},
        })
    return {
        "schemaVersion": 1,
        "kind": "experiment7_universal_bc",
        "staticProfile": f"10d-deck-specialist-static-bc:{archetype}",
        "referenceRoot": str(runtime_root / "reference"),
        "strictPredicate": "is_clean == 1 and float(min_score) > 1000.0",
        "minGameScoreExclusive": 1000.0,
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--strict-manifest", type=Path, required=True)
    parser.add_argument("--archetype", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    local_root = args.local_root.resolve()
    control = args.control_root.resolve() / "profiles" / args.archetype
    runtime = args.runtime_root.resolve()
    acquire(control / "profile.lock")
    state_path = control / "state.json"
    state = {
        "schemaVersion": 1,
        "profile": f"10d-deck-specialist-static-bc:{args.archetype}",
        "pid": os.getpid(),
        "status": "building_immutable_day_shards",
        "createdAt": now(),
        "localRoot": str(local_root),
        "gpu": args.device,
        "batchSize": args.batch_size,
    }
    atomic_json(state_path, state)
    config = load_json(args.config.resolve())
    manifest = load_json(args.strict_manifest.resolve())
    engine_catalog = Path(manifest["engineCatalog"]["path"])
    for index, dataset in enumerate(manifest["datasets"], start=1):
        day = str(dataset["name"])
        target = local_root / "day-shards" / day / args.archetype
        if (target / "SUCCESS").is_file():
            continue
        strict_day = Path(dataset["root"])
        strict_receipt_path = Path(dataset.get("auditReceipt") or strict_day / "audit-receipt.json")
        if not strict_receipt_path.is_file():
            strict_receipt_path = strict_day / "audit-receipt.json"
        strict_receipt = load_json(strict_receipt_path)
        source_catalog = Path(strict_receipt["catalog"]["source"]).parent
        state.update({"day": day, "dayIndex": index, "dayCount": len(manifest["datasets"]), "observedAt": now()})
        atomic_json(state_path, state)
        run([
            str(args.python), "-s", str(runtime / "integration" / "build_static_deck_day_shard.py"),
            "--config", str(args.config.resolve()),
            "--archetype", args.archetype,
            "--strict-day-root", str(strict_day),
            "--source-catalog-dir", str(source_catalog),
            "--engine-catalog", str(engine_catalog),
            "--strict-builder", str(runtime / "integration" / "build_strict_scoregt1000_window.py"),
            "--output", str(target),
        ], local_root / "logs" / f"build-{day}.log")
    sources = specialist_sources(manifest, args.archetype, local_root, runtime)
    sources_path = local_root / "tensordict-sources.json"
    atomic_json(sources_path, sources)
    if args.build_only:
        state.update({
            "status": "immutable_shards_ready",
            "sources": str(sources_path),
            "formalTrainingStarted": False,
            "completedAt": now(),
        })
        atomic_json(control / "completion.json", state)
        atomic_json(state_path, state)
        return
    training = local_root / "training"
    local_initializer = local_root / "initializer.pt"
    initializer_args = (
        ["--initialize-from", str(local_initializer)]
        if local_initializer.is_file()
        else []
    )
    state.update({"status": "starting_async_validator", "sources": str(sources_path), "observedAt": now()})
    atomic_json(state_path, state)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.device)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(runtime / "integration"), existing_pythonpath) if path
    )
    validator_log = (local_root / "logs" / "validator.log").open("a", encoding="utf-8")
    validator = subprocess.Popen([
        str(args.python), "-s", str(runtime / "integration" / "validate_static_deck_bc_async.py"),
        "--config", str(args.config.resolve()), "--sources", str(sources_path),
        "--output-dir", str(training), "--device", "cuda:0", "--batch-size", str(args.batch_size),
        *initializer_args,
    ], stdout=validator_log, stderr=subprocess.STDOUT, env=environment)
    baseline_deadline = time.time() + 3600
    while not (training / "async-validation-report.json").is_file():
        if validator.poll() is not None:
            raise RuntimeError(f"validator exited before baseline: {validator.returncode}")
        if time.time() > baseline_deadline:
            raise TimeoutError("validator baseline did not complete in one hour")
        time.sleep(5)
    trainer_log = (local_root / "logs" / "trainer.log").open("a", encoding="utf-8")
    trainer = subprocess.Popen([
        str(args.python), "-s", str(runtime / "integration" / "train_static_deck_bc_async.py"),
        "--config", str(args.config.resolve()), "--sources", str(sources_path),
        "--output-dir", str(training), "--device", "cuda:0", "--batch-size", str(args.batch_size),
        *initializer_args,
    ], stdout=trainer_log, stderr=subprocess.STDOUT, env=environment)
    state.update({
        "status": "training_with_async_validation",
        "trainerPid": trainer.pid,
        "validatorPid": validator.pid,
        "observedAt": now(),
    })
    atomic_json(state_path, state)
    trainer_rc = trainer.wait()
    validator_rc = validator.wait()
    trainer_log.close()
    validator_log.close()
    best = training / "best_model.pt"
    final = {
        **state,
        "status": "training_complete" if trainer_rc == 0 and validator_rc == 0 and best.is_file() else "training_failed",
        "trainerReturnCode": trainer_rc,
        "validatorReturnCode": validator_rc,
        "bestModel": str(best) if best.is_file() else None,
        "completedAt": now(),
        "staticFrozen": True,
        "ppoUpdatesAllowed": False,
    }
    atomic_json(control / "completion.json", final)
    atomic_json(state_path, final)
    if final["status"] != "training_complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
