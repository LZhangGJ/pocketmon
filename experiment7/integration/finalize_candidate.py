from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "experiment7" / "reference_impl"
TRAINING = REFERENCE / "training"
DATA = REFERENCE / "data_pipeline"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the best sealed-calibration seed, export portable weights, verify parity, and package all decks"
    )
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--selected-decks", type=Path, required=True)
    parser.add_argument("--engine-catalog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    for report_path in sorted(args.training_root.glob("seed-*/model/training_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        checkpoint = report_path.parent / "best_model.pt"
        if not checkpoint.is_file():
            continue
        candidates.append(
            {
                "report": report_path,
                "checkpoint": checkpoint,
                "seed": report.get("seed"),
                "score": float(report["selectedCalibrationMacroExactSemantic"]),
                "selected_epoch": int(report["selectedFineTuneEpoch"]),
            }
        )
    if not candidates:
        raise RuntimeError(f"no completed training candidates under {args.training_root}")
    candidates.sort(key=lambda row: (row["score"], -int(row["selected_epoch"])), reverse=True)
    best = candidates[0]

    args.output_root.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output_root / "best_model.pt"
    shutil.copy2(best["checkpoint"], checkpoint)
    portable = args.output_root / "deck_identity_bc.npz"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("export_portable.py")),
            "--checkpoint", str(checkpoint),
            "--output", str(portable),
        ],
        cwd=ROOT,
        check=True,
    )

    verify_command = [
        sys.executable,
        str(TRAINING / "verify_deck_identity_portable.py"),
        "--checkpoint", str(checkpoint),
        "--portable", str(portable),
        "--decisions-per-source", str(max(1, math.ceil(500 / len(manifest["current_sources"])) )),
        "--output", str(args.output_root / "portable_parity.json"),
    ]
    for source in manifest["current_sources"]:
        verify_command.extend(
            [
                "--source", source["name"], source["features"], source["token_cache"],
                source["sequence_cache"], source["identity_cache"],
            ]
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(TRAINING), str(DATA), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    subprocess.run(verify_command, cwd=TRAINING, env=env, check=True)

    package_root = args.output_root / "packages"
    model_id = f"seed-{best['seed']}-epoch-{best['selected_epoch']}"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("package_agents.py")),
            "--selected-decks", str(args.selected_decks),
            "--portable", str(portable),
            "--engine-catalog", str(args.engine_catalog),
            "--output-root", str(package_root),
            "--model-id", model_id,
        ],
        cwd=ROOT,
        check=True,
    )
    receipt = {
        "schema_version": 1,
        "selection_boundary": "calibration only; chronological holdout remains sealed",
        "candidates": [
            {
                "seed": row["seed"],
                "calibration_macro_exact_semantic": row["score"],
                "selected_epoch": row["selected_epoch"],
                "checkpoint_sha256": sha256_file(row["checkpoint"]),
            }
            for row in candidates
        ],
        "selected": {
            "seed": best["seed"],
            "calibration_macro_exact_semantic": best["score"],
            "selected_epoch": best["selected_epoch"],
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "portable": str(portable.resolve()),
            "portable_sha256": sha256_file(portable),
            "packages": str(package_root.resolve()),
        },
    }
    write_json(args.output_root / "finalization_receipt.json", receipt)
    print(args.output_root / "finalization_receipt.json")


if __name__ == "__main__":
    main()
