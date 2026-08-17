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


def baseline_metrics(report_path: Path) -> tuple[float, float]:
    report = read_json(report_path)
    valid_rows = [
        row
        for row in report.get("epochs", [])
        if isinstance(row.get("validation"), dict)
    ]
    if not valid_rows:
        if "baselineExactSemantic" in report and "baselineValueBrier" in report:
            return (
                float(report["baselineExactSemantic"]),
                float(report["baselineValueBrier"]),
            )
        raise ValueError(f"baseline report has no validation metrics: {report_path}")
    best = report.get("best", {})
    best_epoch = best.get("epoch")
    row = next(
        (row for row in valid_rows if best_epoch is not None and int(row["epoch"]) == int(best_epoch)),
        None,
    )
    if row is None:
        row = max(valid_rows, key=lambda item: float(item["validation"]["exactSemantic"]))
    validation = row["validation"]
    return float(validation["exactSemantic"]), float(validation["valueBrier"])


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-semantic-delta", type=float, default=0.002)
    parser.add_argument("--max-brier-increase", type=float, default=0.005)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()

    sources = read_json(args.sources.resolve())
    vendor = core.setup_vendor(Path(sources["referenceRoot"]))
    core.seed_everything(20260813)
    device = core.device_from_arg(args.device)
    shards = core.prepare_shards(sources, vendor, 0.05, 0, 0)
    output = args.output_dir.resolve()
    queue = output / "validation-queue"
    results = output / "validation-results"
    control = output / "control"
    for path in (queue, results, control):
        path.mkdir(parents=True, exist_ok=True)

    baseline_score, previous_brier = baseline_metrics(args.baseline_report.resolve())
    # Checkpoint selection and early-stop significance are deliberately separate.
    # A small raw improvement is still the best artifact to screen, even when it
    # is below the delta required to reset patience.
    raw_best_score = baseline_score
    raw_best_checkpoint = args.baseline_checkpoint.resolve()
    early_stop_reference_score = baseline_score
    stagnant = 0
    brier_regressions = 0
    report = {
        "schemaVersion": 1,
        "mode": "persistent_async_validation_worker",
        "createdAt": utc_now(),
        "device": str(device),
        "baselineReport": str(args.baseline_report.resolve()),
        "baselineExactSemantic": baseline_score,
        "baselineValueBrier": previous_brier,
        "epochs": [],
    }
    report_path = output / "async-validation-report.json"
    atomic_copy(raw_best_checkpoint, output / "best_model.pt")
    write_json(report_path, report)
    model = None
    while True:
        jobs = sorted(queue.glob("epoch_*.json"))
        job_path = next((p for p in jobs if not (results / p.name).is_file()), None)
        if job_path is None:
            train_report = read_json(output / "async-training-report.json") if (output / "async-training-report.json").is_file() else {}
            if train_report.get("completedAt"):
                break
            time.sleep(args.poll_seconds)
            continue
        job = read_json(job_path)
        epoch = int(job["epoch"])
        checkpoint = Path(job["checkpoint"])
        payload = core.load_checkpoint(checkpoint, device)
        config = vendor["UniversalDeckModelConfig"](**payload["config"])
        if model is None:
            model = vendor["UniversalDeckTransformerPolicy"](config).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        rows = []
        for shard in shards:
            metrics = core.evaluate(vendor, model, shard["bundle"], shard["validation"], device, args.batch_size)
            rows.append((shard["name"], metrics))
            print(json.dumps({"stage": "validation_shard", "epoch": epoch, "shard": shard["name"], **metrics}), flush=True)
        validation = core.combine_validation_metrics(rows)
        score = float(validation["exactSemantic"])
        brier = float(validation["valueBrier"])
        raw_improved = score > raw_best_score
        if raw_improved:
            raw_best_score = score
            raw_best_checkpoint = checkpoint
            atomic_copy(checkpoint, output / "best_model.pt")
        meaningful_improvement = score >= early_stop_reference_score + args.min_semantic_delta
        if meaningful_improvement:
            early_stop_reference_score = score
            stagnant = 0
        else:
            stagnant += 1
        if brier > previous_brier + args.max_brier_increase:
            brier_regressions += 1
        else:
            brier_regressions = 0
        previous_brier = brier
        row = {
            "epoch": epoch, "checkpoint": str(checkpoint), "validation": validation,
            # Keep `improved` backward-compatible: it means a statistically/
            # operationally meaningful improvement for early stopping.
            "improved": meaningful_improvement,
            "meaningfulImprovement": meaningful_improvement,
            "rawImproved": raw_improved,
            "bestExactSemantic": raw_best_score,
            "bestCheckpoint": str(raw_best_checkpoint),
            "earlyStopReferenceExactSemantic": early_stop_reference_score,
            "stagnantEpochs": stagnant, "brierRegressionEpochs": brier_regressions,
            "completedAt": utc_now(),
        }
        write_json(results / job_path.name, row)
        report["epochs"].append(row)
        report["best"] = {"score": raw_best_score, "checkpoint": str(raw_best_checkpoint)}
        report["earlyStopReferenceExactSemantic"] = early_stop_reference_score
        write_json(report_path, report)
        print(json.dumps({"stage": "validation_complete", **row}), flush=True)
        if stagnant >= args.patience or brier_regressions >= args.patience:
            stop = {
                "schemaVersion": 1, "status": "stop_requested", "requestedAt": utc_now(),
                "epoch": epoch,
                "reason": "semantic_plateau" if stagnant >= args.patience else "value_brier_regression",
                "bestExactSemantic": raw_best_score,
                "bestCheckpoint": str(raw_best_checkpoint),
                "earlyStopReferenceExactSemantic": early_stop_reference_score,
            }
            write_json(control / "stop-request.json", stop)
            report["stop"] = stop
            report["completedAt"] = utc_now()
            write_json(report_path, report)
            break


if __name__ == "__main__":
    main()
