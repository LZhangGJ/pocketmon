from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch

import train_universal_bc as core
from common import read_json, utc_now, write_json
from static_deck_bc_common import load_json
from train_static_deck_bc_async import validate_static_sources


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def evaluate_checkpoint(vendor, shards, checkpoint: Path, device, batch_size: int, model=None):
    payload = core.load_checkpoint(checkpoint, device)
    model_config = vendor["UniversalDeckModelConfig"](**payload["config"])
    if model is None:
        model = vendor["UniversalDeckTransformerPolicy"](model_config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    rows = []
    for shard in shards:
        metrics = core.evaluate(vendor, model, shard["bundle"], shard["validation"], device, batch_size)
        rows.append((shard["name"], metrics))
    return core.combine_validation_metrics(rows), model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    sources = validate_static_sources(args.sources.resolve(), config)
    vendor = core.setup_vendor(Path(sources["referenceRoot"]))
    core.seed_everything(20260815)
    device = core.device_from_arg(args.device)
    shards = core.prepare_shards(sources, vendor, 0.05, 0, 0)
    output = args.output_dir.resolve()
    queue, results, control = (output / "validation-queue", output / "validation-results", output / "control")
    for path in (queue, results, control):
        path.mkdir(parents=True, exist_ok=True)
    baseline_checkpoint = args.initialize_from or Path(config["training"]["initializer"])
    baseline, model = evaluate_checkpoint(vendor, shards, baseline_checkpoint, device, args.batch_size)
    best_score = float(baseline["exactSemantic"])
    best_checkpoint = baseline_checkpoint
    existing_rows = []
    for result_path in sorted(results.glob("epoch_*.json")):
        try:
            row = read_json(result_path)
            score = float(row["validation"]["exactSemantic"])
        except (KeyError, TypeError, ValueError):
            continue
        existing_rows.append(row)
        if score > best_score:
            best_score = score
            best_checkpoint = Path(row["checkpoint"])
    preserved_best = output / "best_model.pt"
    if preserved_best.is_file():
        best_checkpoint = preserved_best
    else:
        atomic_copy(best_checkpoint, preserved_best)
    report = {
        "schemaVersion": 1,
        "mode": "static_deck_bc_async_validation",
        "createdAt": utc_now(),
        "selectionMetric": "exactSemantic",
        "patience": int(config["training"]["patience"]),
        "minEpochs": int(config["training"]["minEpochs"]),
        "maxEpochs": int(config["training"]["maxEpochs"]),
        "baselineCheckpoint": str(baseline_checkpoint),
        "baselineValidation": baseline,
        "epochs": existing_rows,
    }
    if existing_rows:
        report["best"] = {"score": best_score, "checkpoint": str(best_checkpoint)}
    write_json(output / "async-validation-report.json", report)
    while True:
        jobs = sorted(queue.glob("epoch_*.json"))
        job_path = next((path for path in jobs if not (results / path.name).is_file()), None)
        if job_path is None:
            training_report = read_json(output / "async-training-report.json") if (output / "async-training-report.json").is_file() else {}
            if training_report.get("completedAt"):
                break
            time.sleep(args.poll_seconds)
            continue
        job = read_json(job_path)
        epoch = int(job["epoch"])
        checkpoint = Path(job["checkpoint"])
        validation, model = evaluate_checkpoint(vendor, shards, checkpoint, device, args.batch_size, model)
        score = float(validation["exactSemantic"])
        improved = score > best_score
        if improved:
            best_score = score
            best_checkpoint = checkpoint
            atomic_copy(checkpoint, output / "best_model.pt")
        row = {
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "validation": validation,
            "improved": improved,
            "bestExactSemantic": best_score,
            "bestCheckpoint": str(best_checkpoint),
            "completedAt": utc_now(),
        }
        write_json(results / job_path.name, row)
        report["epochs"].append(row)
        report["best"] = {"score": best_score, "checkpoint": str(best_checkpoint)}
        write_json(output / "async-validation-report.json", report)
        print(json.dumps({"stage": "validation_complete", **row}), flush=True)
        if epoch >= int(config["training"]["minEpochs"]) and not improved:
            stop = {
                "schemaVersion": 1,
                "status": "stop_requested",
                "reason": "exact_semantic_plateau",
                "epoch": epoch,
                "bestExactSemantic": best_score,
                "bestCheckpoint": str(best_checkpoint),
                "requestedAt": utc_now(),
            }
            write_json(control / "stop-request.json", stop)
            report["stop"] = stop
            report["completedAt"] = utc_now()
            write_json(output / "async-validation-report.json", report)
            break


if __name__ == "__main__":
    main()
