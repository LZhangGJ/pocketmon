from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import train_universal_bc as core
from common import read_json, utc_now, write_json


def baseline_metrics(path: Path) -> tuple[float, float]:
    report = read_json(path)
    rows = [row for row in report.get("epochs", []) if isinstance(row.get("validation"), dict)]
    if not rows:
        return float(report["baselineExactSemantic"]), float(report["baselineValueBrier"])
    best_epoch = report.get("best", {}).get("epoch")
    row = next((row for row in rows if best_epoch is not None and int(row["epoch"]) == int(best_epoch)), None)
    row = row or max(rows, key=lambda item: float(item["validation"]["exactSemantic"]))
    return float(row["validation"]["exactSemantic"]), float(row["validation"]["valueBrier"])


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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--minimum-epochs", type=int, default=6)
    parser.add_argument("--min-semantic-delta", type=float, default=0.001)
    parser.add_argument("--max-brier-increase", type=float, default=0.005)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()

    sources = read_json(args.sources.resolve())
    vendor = core.setup_vendor(Path(sources["referenceRoot"]))
    core.seed_everything(20260815)
    device = core.device_from_arg(args.device)
    shards = core.prepare_shards(sources, vendor, 0.05, 0, 0)
    output = args.output_dir.resolve()
    queue, results, control = output / "validation-queue", output / "validation-results", output / "control"
    for path in (queue, results, control):
        path.mkdir(parents=True, exist_ok=True)
    baseline_score, previous_brier = baseline_metrics(args.baseline_report.resolve())
    raw_best_score, raw_best_checkpoint = -1.0, None
    early_reference, stagnant, brier_regressions = baseline_score, 0, 0
    report = {
        "schemaVersion": 1,
        "mode": "scaled_persistent_async_validation_worker",
        "createdAt": utc_now(),
        "device": str(device),
        "baselineExactSemantic": baseline_score,
        "baselineValueBrier": previous_brier,
        "minimumEpochs": args.minimum_epochs,
        "epochs": [],
    }
    report_path = output / "async-validation-report.json"
    write_json(report_path, report)
    model = None
    while True:
        jobs = sorted(queue.glob("epoch_*.json"))
        job_path = next((path for path in jobs if not (results / path.name).is_file()), None)
        if job_path is None:
            training = read_json(output / "async-training-report.json") if (output / "async-training-report.json").is_file() else {}
            if training.get("completedAt"):
                break
            time.sleep(args.poll_seconds)
            continue
        job = read_json(job_path)
        epoch, checkpoint = int(job["epoch"]), Path(job["checkpoint"])
        payload = core.load_checkpoint(checkpoint, device)
        config = vendor["UniversalDeckModelConfig"](**payload["config"])
        if model is None:
            model = vendor["UniversalDeckTransformerPolicy"](config).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        rows = []
        for shard in shards:
            metrics = core.evaluate(vendor, model, shard["bundle"], shard["validation"], device, args.batch_size)
            rows.append((shard["name"], metrics))
            print({"stage": "validation_shard", "epoch": epoch, "shard": shard["name"], **metrics}, flush=True)
        validation = core.combine_validation_metrics(rows)
        score, brier = float(validation["exactSemantic"]), float(validation["valueBrier"])
        raw_improved = score > raw_best_score
        if raw_improved:
            raw_best_score, raw_best_checkpoint = score, checkpoint
            atomic_copy(checkpoint, output / "best_model.pt")
        meaningful = score >= early_reference + args.min_semantic_delta
        if meaningful:
            early_reference, stagnant = score, 0
        elif epoch >= args.minimum_epochs:
            stagnant += 1
        if epoch >= args.minimum_epochs and brier > previous_brier + args.max_brier_increase:
            brier_regressions += 1
        else:
            brier_regressions = 0
        previous_brier = brier
        row = {
            "epoch": epoch, "checkpoint": str(checkpoint), "validation": validation,
            "rawImproved": raw_improved, "meaningfulImprovement": meaningful,
            "bestExactSemantic": raw_best_score, "bestCheckpoint": str(raw_best_checkpoint),
            "earlyStopReferenceExactSemantic": early_reference,
            "stagnantEpochs": stagnant, "brierRegressionEpochs": brier_regressions,
            "completedAt": utc_now(),
        }
        write_json(results / job_path.name, row)
        report["epochs"].append(row)
        report["best"] = {"score": raw_best_score, "checkpoint": str(raw_best_checkpoint)}
        write_json(report_path, report)
        print({"stage": "validation_complete", **row}, flush=True)
        if epoch >= args.minimum_epochs and (stagnant >= args.patience or brier_regressions >= args.patience):
            stop = {
                "schemaVersion": 1, "status": "stop_requested", "requestedAt": utc_now(),
                "epoch": epoch,
                "reason": "semantic_plateau" if stagnant >= args.patience else "value_brier_regression",
                "bestExactSemantic": raw_best_score, "bestCheckpoint": str(raw_best_checkpoint),
            }
            write_json(control / "stop-request.json", stop)
            report["stop"], report["completedAt"] = stop, utc_now()
            write_json(report_path, report)
            break


if __name__ == "__main__":
    main()
