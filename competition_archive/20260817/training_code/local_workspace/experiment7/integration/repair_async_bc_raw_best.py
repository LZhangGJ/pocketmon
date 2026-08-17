from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def repair(output: Path, baseline_checkpoint: Path) -> dict:
    report_path = output / "async-validation-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidates = [
        {
            "score": float(report["baselineExactSemantic"]),
            "checkpoint": str(baseline_checkpoint.resolve()),
            "epoch": "baseline",
        }
    ]
    for result_path in sorted((output / "validation-results").glob("epoch_*.json")):
        row = json.loads(result_path.read_text(encoding="utf-8"))
        candidates.append(
            {
                "score": float(row["validation"]["exactSemantic"]),
                "checkpoint": str(Path(row["checkpoint"]).resolve()),
                "epoch": int(row["epoch"]),
            }
        )
    raw_best = max(candidates, key=lambda row: (row["score"], str(row["epoch"])))
    checkpoint = Path(raw_best["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    previous = dict(report.get("best", {}))
    if previous and previous != raw_best:
        report["bestByMeaningfulDelta"] = previous
    report["best"] = raw_best
    report["selectionSemantics"] = "raw_score_for_artifact; min_delta_only_resets_early_stop_patience"
    if report.get("stop"):
        report["stop"]["bestExactSemantic"] = raw_best["score"]
        report["stop"]["bestCheckpoint"] = raw_best["checkpoint"]
    atomic_copy(checkpoint, output / "best_model.pt")
    write_json(report_path, report)
    receipt = {
        "schemaVersion": 1,
        "repairedAt": datetime.now(timezone.utc).isoformat(),
        "previousSelection": previous,
        "rawBestSelection": raw_best,
        "artifact": str((output / "best_model.pt").resolve()),
    }
    write_json(output / "raw-best-selection-receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair async BC selection after min-delta/raw-best conflation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(repair(args.output.resolve(), args.baseline_checkpoint.resolve())))


if __name__ == "__main__":
    main()
