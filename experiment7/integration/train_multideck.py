from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from common import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "experiment7" / "reference_impl"
TRAINING = REFERENCE / "training"
DATA_PIPELINE = REFERENCE / "data_pipeline"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the accepted Experiment 7 K-deck training implementation")
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--pretrain-batch", type=int, default=128)
    parser.add_argument("--finetune-batch-per-deck", type=int, default=48)
    parser.add_argument("--tiny-decisions", type=int, default=0)
    parser.add_argument("--baseline-checkpoint", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    current = manifest.get("current_sources") or []
    if len(current) < 2:
        raise RuntimeError("Experiment 7 multideck training requires at least two current sources")
    pretrain = manifest["pretrain"]
    required = [
        Path(pretrain["features"]),
        Path(pretrain["token_cache"]),
        Path(pretrain["sequence_cache"]),
        Path(pretrain["identity_cache"]),
        Path(manifest["catalog"]["path"]),
        Path(manifest["class_map"]["path"]),
    ]
    for source in current:
        required.extend(
            [
                Path(source["features"]),
                Path(source["token_cache"]),
                Path(source["sequence_cache"]),
                Path(source["identity_cache"]),
            ]
        )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(TRAINING / "train_multideck_identity.py"),
        "--pretrain-features", pretrain["features"],
        "--pretrain-cache", pretrain["token_cache"],
        "--pretrain-sequence-cache", pretrain["sequence_cache"],
        "--pretrain-identity-cache", pretrain["identity_cache"],
        "--pretrain-catalog", manifest["catalog"]["path"],
        "--class-map", manifest["class_map"]["path"],
        "--output-dir", str(args.output_dir),
        "--pretrain-epochs", str(args.pretrain_epochs),
        "--finetune-epochs", str(args.finetune_epochs),
        "--pretrain-batch", str(args.pretrain_batch),
        "--finetune-batch-per-deck", str(args.finetune_batch_per_deck),
        "--seed", str(args.seed),
        "--d-model", "128",
        "--layers", "3",
        "--heads", "4",
        "--ff-dim", "384",
        "--pretrain-lr", "0.0003",
        "--finetune-lr", "0.0001",
        "--opponent-loss-weight", "0.05",
    ]
    for source in current:
        command.extend(
            [
                "--current-source",
                source["name"],
                source["features"],
                source["token_cache"],
                source["sequence_cache"],
                source["identity_cache"],
                str(source["calibration_episode_count"]),
            ]
        )
    if args.tiny_decisions:
        command.extend(["--tiny-decisions", str(args.tiny_decisions)])
    if args.baseline_checkpoint:
        command.extend(["--baseline-checkpoint", str(args.baseline_checkpoint)])

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(TRAINING), str(DATA_PIPELINE), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    command_path = args.output_dir / "command.json"
    write_json(
        command_path,
        {
            "schema_version": 1,
            "command": command,
            "seed": args.seed,
            "dataset_manifest": str(args.dataset_manifest.resolve()),
            "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
            "holdout_opened": False,
        },
    )
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=TRAINING, env=env, check=True)
    checkpoint = args.output_dir / "best_model.pt"
    report = args.output_dir / "training_report.json"
    if not checkpoint.is_file() or not report.is_file():
        raise RuntimeError("training did not produce best_model.pt and training_report.json")
    write_json(
        args.output_dir / "run_receipt.json",
        {
            "schema_version": 1,
            "status": "trained_holdout_sealed",
            "seed": args.seed,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "training_report": str(report.resolve()),
            "training_report_sha256": sha256_file(report),
            "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
        },
    )
    print(args.output_dir / "run_receipt.json")


if __name__ == "__main__":
    main()
