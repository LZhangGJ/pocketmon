from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


LOSS_FIELDS = {
    "validation_total_loss": ("validation_loss", "loss"),
    "validation_policy_loss": ("validation_loss", "policy_loss"),
    "validation_value_loss": ("validation_loss", "value_loss"),
}
VALIDATION_FIELDS = (
    "sequence_exact_match",
    "set_exact_match",
    "candidate_precision",
    "candidate_recall",
    "single_select_accuracy",
    "empty_action_accuracy",
    "multi_select_accuracy",
    "decode_legal_rate",
)


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return True


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def aggregate_reports(reports: list[dict[str, Any]], expected_seeds: list[int]) -> dict[str, Any]:
    if not reports:
        raise ValueError("no metric reports")
    if not all(_finite_tree(report) for report in reports):
        raise ValueError("non-finite metric in report")
    fingerprints = {report["provenance"]["experiment_fingerprint"] for report in reports}
    commits = {report["provenance"]["git_sha"] for report in reports}
    input_hashes = {report["provenance"]["input_sha256"] for report in reports}
    experiment_ids = {report["experiment_id"] for report in reports}
    if len(fingerprints) != 1 or len(commits) != 1 or len(input_hashes) != 1 or len(experiment_ids) != 1:
        raise ValueError("cannot aggregate different experiment fingerprints, commits, inputs, or arms")
    seeds = [int(report["actual_config"]["seed"]) for report in reports]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed report")
    missing = sorted(set(expected_seeds) - set(seeds))
    unexpected = sorted(set(seeds) - set(expected_seeds))
    config_mismatches = {
        seed: report["actual_config"].get("config_mismatches", [])
        for seed, report in zip(seeds, reports)
        if not report["actual_config"].get("config_matched", False)
    }
    invalid_actions = sum(int(report["validation"]["invalid_actions"]) for report in reports)
    unsupported_rows = sum(int(report["dataset_audit"]["unsupported_rows"]) for report in reports)
    skipped_rows = sum(int(report["dataset_audit"]["skipped_rows"]) for report in reports)
    legal_min = min(float(report["validation"]["decode_legal_rate"]) for report in reports)
    failures = [failure for report in reports for failure in report.get("failures", [])]
    status = "completed_formal"
    if missing or unexpected or config_mismatches:
        status = "partial_formal"
    if invalid_actions or unsupported_rows or skipped_rows or legal_min != 1.0 or failures:
        status = "failed"

    aggregate: dict[str, Any] = {}
    for output_name, (section, field) in LOSS_FIELDS.items():
        aggregate[output_name] = _stats([float(report[section][field]) for report in reports])
    for field in VALIDATION_FIELDS:
        aggregate[field] = _stats([float(report["validation"][field]) for report in reports])
    aggregate["best_epoch"] = _stats([float(report["best_epoch"]) for report in reports])
    aggregate["runtime_seconds"] = _stats([float(report["runtime_seconds"]) for report in reports])
    aggregate["peak_ram_mb"] = _stats([float(report["peak_ram_mb"]) for report in reports])
    aggregate["peak_vram_mb"] = _stats([float(report["peak_vram_mb"]) for report in reports])

    per_seed = []
    for report in sorted(reports, key=lambda item: int(item["actual_config"]["seed"])):
        history = report["training_state"]["history"]
        late = history[-5:]
        epoch30 = next((record for record in history if int(record["epoch"]) == 30), None)
        per_seed.append({
            "seed": int(report["actual_config"]["seed"]),
            "best_epoch": int(report["best_epoch"]),
            "epochs_completed": int(report["epochs_completed"]),
            "validation_loss": report["validation_loss"],
            "validation": report["validation"],
            "epoch_30_validation": epoch30["validation"] if epoch30 else None,
            "last_five_epochs": [
                {"epoch": int(record["epoch"]), "train_loss": record["train"]["loss"], "validation_loss": record["validation"]["loss"]}
                for record in late
            ],
            "late_validation_loss_change": (
                float(late[-1]["validation"]["loss"]) - float(late[0]["validation"]["loss"])
                if len(late) > 1 else 0.0
            ),
            "checkpoint": report["checkpoint"],
            "runtime_seconds": report["runtime_seconds"],
            "peak_ram_mb": report["peak_ram_mb"],
            "peak_vram_mb": report["peak_vram_mb"],
            "raw_metrics": report.get("raw_metrics"),
        })

    first = reports[0]
    return {
        "experiment_id": first["experiment_id"],
        "status": status,
        "provenance": {
            "git_sha": next(iter(commits)),
            "input_sha256": next(iter(input_hashes)),
            "experiment_fingerprint": next(iter(fingerprints)),
            "dirty": any(bool(report["provenance"]["dirty"]) for report in reports),
        },
        "planned_config": first["planned_config"],
        "achieved_seeds": sorted(seeds),
        "missing_seeds": missing,
        "unexpected_seeds": unexpected,
        "config_mismatches": config_mismatches,
        "split": first["split"],
        "dataset_audit": first["dataset_audit"],
        "aggregate": aggregate,
        "legality": {
            "decode_legal_rate_min": legal_min,
            "invalid_actions": invalid_actions,
            "unsupported_rows": unsupported_rows,
            "skipped_rows": skipped_rows,
        },
        "per_seed": per_seed,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate matching RL-BC seed reports")
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[17, 42, 20260720])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports = []
    for name in args.metrics:
        report = json.loads(Path(name).read_text(encoding="utf-8"))
        report["raw_metrics"] = name
        reports.append(report)
    aggregate = aggregate_reports(reports, args.expected_seeds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": aggregate["status"],
        "achieved_seeds": aggregate["achieved_seeds"],
        "missing_seeds": aggregate["missing_seeds"],
        "fingerprint": aggregate["provenance"]["experiment_fingerprint"],
    }))


if __name__ == "__main__":
    main()
