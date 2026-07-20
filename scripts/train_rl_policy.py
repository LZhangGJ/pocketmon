from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.bc import (
    SPLIT_SEED,
    TrajectoryDataset,
    build_split_manifest,
    evaluate_decoding,
    load_replay_dataset,
    make_loader,
    run_epoch,
)
from rl.model import MaskedPointerActorCritic


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path: Path, model: MaskedPointerActorCritic, optimizer: torch.optim.Optimizer, epoch: int, config: dict[str, Any], metrics: dict[str, Any], input_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
        "config": config, "metrics": metrics, "input_sha256": input_sha256,
        "hidden_dim": model.hidden_dim, "git_sha": git_sha(),
    }, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_run_csv(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "experiment_id", "seed", "status", "epochs_completed", "best_epoch", "best_validation_loss",
        "sequence_exact_match", "set_exact_match", "empty_action_accuracy", "multi_select_accuracy",
        "decode_legal_rate", "invalid_actions", "unsupported_rows", "skipped_rows", "checkpoint",
        "checkpoint_sha256", "runtime_seconds", "peak_ram_mb", "peak_vram_mb", "git_sha",
    ]
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = [row for row in csv.DictReader(handle) if not (row["experiment_id"] == str(record["experiment_id"]) and row["seed"] == str(record["seed"]))]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(existing); writer.writerow({key: record.get(key) for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="RL-BC-001 stateless masked behavior cloning")
    parser.add_argument("--input", default="data/processed/public_replay_v1.jsonl.gz")
    parser.add_argument("--experiment-id", default="RL-BC-001")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-dir", default="checkpoints/rl_bc_001")
    parser.add_argument("--metrics-output", default="results/rl_bc_001_metrics.json")
    parser.add_argument("--split-output", default="results/rl_bc_001_split.json")
    parser.add_argument("--runs-output", default="results/rl_bc_001_runs.csv")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-batches", type=int, default=0, help="Smoke-only batch limit; marks the run partial")
    args = parser.parse_args()
    start = time.perf_counter()
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    rows, audit = load_replay_dataset(Path(args.input))
    manifest, train_ids, validation_ids = build_split_manifest(rows, args.validation_fraction, args.split_seed)
    if train_ids & validation_ids or manifest["episode_overlap"]:
        raise AssertionError("episode leakage detected")
    if manifest["rows"] != audit["readable_rows"]:
        raise AssertionError("split row total mismatch")
    manifest.update({"input_sha256": audit["input_sha256"], "input_bytes": audit["input_bytes"]})
    write_json(Path(args.split_output), manifest)
    train = TrajectoryDataset([row for row in rows if row["episode_id"] in train_ids])
    validation = TrajectoryDataset([row for row in rows if row["episode_id"] in validation_ids])
    train_loader = make_loader(train, args.batch_size, True, args.seed)
    validation_loader = make_loader(validation, args.batch_size, False, args.seed)
    model = MaskedPointerActorCritic(args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
    config = vars(args).copy()
    config.update({"git_sha": git_sha(), "device_resolved": str(device), "stateless": True, "decoder": "masked_autoregressive_pointer_with_stop", "policy_rows": "winner_only", "value_rows": "both_players"})
    checkpoint_dir = Path(args.checkpoint_dir)
    best_path = checkpoint_dir / f"seed_{args.seed}_best.pt"
    last_path = checkpoint_dir / f"seed_{args.seed}_last.pt"
    history = []
    best_loss = float("inf"); best_epoch = 0; stale = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, args.max_batches)
        validation_metrics = run_epoch(model, validation_loader, device, None, args.max_batches)
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(record); print(json.dumps(record), flush=True)
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]; best_epoch = epoch; stale = 0
            save_checkpoint(best_path, model, optimizer, epoch, config, record, audit["input_sha256"])
        else:
            stale += 1
        save_checkpoint(last_path, model, optimizer, epoch, config, record, audit["input_sha256"])
        if not args.max_batches and stale >= args.patience:
            break
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    final_loss = run_epoch(model, validation_loader, device, max_batches=args.max_batches)
    decoding = evaluate_decoding(model, validation_loader, device, max_batches=args.max_batches)
    if decoding["decode_legal_rate"] != 1.0 or decoding["invalid_actions"] != 0:
        raise RuntimeError("validation decode legality gate failed")
    runtime = time.perf_counter() - start
    checkpoint_hash = checkpoint_sha256(best_path)
    result = {
        "experiment_id": args.experiment_id,
        "status": "partial" if args.max_batches else "completed",
        "partial_run": bool(args.max_batches),
        "config": config,
        "dataset_audit": audit,
        "split": {key: manifest[key] for key in ("split_seed", "validation_fraction", "train_episodes", "validation_episodes", "episode_overlap", "rows", "policy_rows", "value_rows")},
        "train_rows": len(train), "validation_rows": len(validation),
        "epochs_completed": history[-1]["epoch"], "best_epoch": best_epoch,
        "best_validation_loss": best_loss, "validation_loss": final_loss,
        "validation": decoding, "history": history,
        "checkpoint": {"path": str(best_path), "sha256": checkpoint_hash, "bytes": best_path.stat().st_size},
        "runtime_seconds": runtime,
        "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0,
        "failures": [],
    }
    write_json(Path(args.metrics_output), result)
    if not args.max_batches:
        append_run_csv(Path(args.runs_output), {
            "experiment_id": args.experiment_id, "seed": args.seed, "status": result["status"],
            "epochs_completed": result["epochs_completed"], "best_epoch": best_epoch, "best_validation_loss": best_loss,
            "sequence_exact_match": decoding["sequence_exact_match"], "set_exact_match": decoding["set_exact_match"],
            "empty_action_accuracy": decoding["empty_action_accuracy"], "multi_select_accuracy": decoding["multi_select_accuracy"],
            "decode_legal_rate": decoding["decode_legal_rate"], "invalid_actions": decoding["invalid_actions"],
            "unsupported_rows": audit["unsupported_rows"], "skipped_rows": audit["skipped_rows"],
            "checkpoint": best_path, "checkpoint_sha256": checkpoint_hash, "runtime_seconds": runtime,
            "peak_ram_mb": result["peak_ram_mb"], "peak_vram_mb": result["peak_vram_mb"], "git_sha": config["git_sha"],
        })
    print(json.dumps({key: result[key] for key in ("status", "train_rows", "validation_rows", "best_epoch", "runtime_seconds", "checkpoint")}), flush=True)


# Backward-compatible imports used by existing tests.
from rl.bc import TrajectoryDataset, split_by_episode  # noqa: E402,F401


if __name__ == "__main__":
    main()
