from __future__ import annotations

import argparse
import copy
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

from rl.bc import SPLIT_SEED, TrajectoryDataset, build_split_manifest, evaluate_decoding, load_replay_dataset, make_loader, run_epoch
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
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_status_porcelain() -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)


def assert_formal_worktree_clean(max_batches: int) -> str:
    dirty = git_status_porcelain()
    if not max_batches and dirty:
        raise RuntimeError(f"formal training requires a clean worktree; git status --porcelain:\n{dirty}")
    return dirty


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_training_state() -> dict[str, Any]:
    return {
        "best_loss": float("inf"), "best_epoch": 0, "stale": 0,
        "history": [], "partial_history": [], "completed_epochs": 0,
        "optimizer_steps": 0, "best_checkpoint_path": None,
    }


def restore_training_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if "training_state" not in checkpoint:
        raise ValueError("checkpoint lacks complete training_state and cannot be resumed safely")
    state = copy.deepcopy(checkpoint["training_state"])
    required = {"best_loss", "best_epoch", "stale", "history", "partial_history", "completed_epochs", "optimizer_steps", "best_checkpoint_path"}
    missing = required - state.keys()
    if missing:
        raise ValueError(f"checkpoint training_state missing: {sorted(missing)}")
    return state


def update_training_state(state: dict[str, Any], record: dict[str, Any], *, full_epoch: bool, optimizer_steps: int, best_path: Path) -> bool:
    """Update resumable state and return whether the formal best must be replaced."""

    state["optimizer_steps"] += int(optimizer_steps)
    if not full_epoch:
        state["partial_history"].append(record)
        return False
    epoch = int(record["epoch"])
    state["completed_epochs"] = epoch
    state["history"].append(record)
    validation_loss = float(record["validation"]["loss"])
    if validation_loss < float(state["best_loss"]):
        state["best_loss"] = validation_loss
        state["best_epoch"] = epoch
        state["stale"] = 0
        state["best_checkpoint_path"] = str(best_path)
        return True
    state["stale"] += 1
    return False


def save_checkpoint(path: Path, model: MaskedPointerActorCritic, optimizer: torch.optim.Optimizer, training_state: dict[str, Any], config: dict[str, Any], metrics: dict[str, Any], input_sha256: str, code_commit: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "training_state": copy.deepcopy(training_state), "config": config, "metrics": metrics,
        "input_sha256": input_sha256, "hidden_dim": model.hidden_dim, "git_sha": code_commit,
    }, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


RUN_FIELDS = [
    "experiment_id", "seed", "status", "epochs_completed", "best_epoch", "best_validation_loss",
    "sequence_exact_match", "set_exact_match", "empty_action_accuracy", "multi_select_accuracy",
    "decode_legal_rate", "invalid_actions", "unsupported_rows", "skipped_rows", "checkpoint",
    "checkpoint_sha256", "runtime_seconds", "peak_ram_mb", "peak_vram_mb", "git_sha", "dirty",
    "planned_seeds", "missing_seeds", "planned_epochs", "actual_epochs",
]


def read_run_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_run_csv(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = [row for row in read_run_rows(path) if not (row["experiment_id"] == str(record["experiment_id"]) and row["seed"] == str(record["seed"]))]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader(); writer.writerows(existing); writer.writerow({key: record.get(key) for key in RUN_FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser(description="RL-BC-001 stateless masked behavior cloning")
    parser.add_argument("--input", default="data/processed/public_replay_v1.jsonl.gz")
    parser.add_argument("--experiment-id", default="RL-BC-001")
    parser.add_argument("--planned-config", default="configs/rl_bc_001.json")
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
    parser.add_argument("--max-batches", type=int, default=0, help="Smoke-only limit; never completes an epoch")
    args = parser.parse_args()

    start = time.perf_counter()
    initial_dirty = assert_formal_worktree_clean(args.max_batches)
    code_commit = git_sha()
    planned_config = json.loads(Path(args.planned_config).read_text(encoding="utf-8"))
    planned_seeds = [int(seed) for seed in planned_config["training"]["formal_seeds"]]
    planned_epochs = int(planned_config["training"]["epochs"])
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats()

    rows, audit = load_replay_dataset(Path(args.input))
    manifest, train_ids, validation_ids = build_split_manifest(rows, args.validation_fraction, args.split_seed)
    if train_ids & validation_ids or manifest["episode_overlap"]:
        raise AssertionError("episode leakage detected")
    if manifest["rows"] != audit["readable_rows"]:
        raise AssertionError("split row total mismatch")
    manifest.update({"input_sha256": audit["input_sha256"], "input_bytes": audit["input_bytes"], "code_commit": code_commit, "dirty_at_start": bool(initial_dirty)})
    write_json(Path(args.split_output), manifest)
    train = TrajectoryDataset([row for row in rows if row["episode_id"] in train_ids])
    validation = TrajectoryDataset([row for row in rows if row["episode_id"] in validation_ids])
    train_loader = make_loader(train, args.batch_size, True, args.seed)
    validation_loader = make_loader(validation, args.batch_size, False, args.seed)
    model = MaskedPointerActorCritic(args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    state = new_training_state()
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("input_sha256") != audit["input_sha256"]:
            raise ValueError("resume input SHA does not match checkpoint")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        state = restore_training_state(checkpoint)

    actual_config = vars(args).copy()
    actual_config.update({
        "code_commit": code_commit, "dirty_at_start": bool(initial_dirty), "input_sha256": audit["input_sha256"],
        "device_resolved": str(device), "stateless": True, "decoder": "masked_autoregressive_pointer_with_stop",
        "policy_rows": "winner_only", "value_rows": "both_players",
    })
    checkpoint_dir = Path(args.checkpoint_dir)
    best_path = checkpoint_dir / f"seed_{args.seed}_best.pt"
    last_path = checkpoint_dir / f"seed_{args.seed}_last.pt"
    full_epochs_this_run = 0
    last_record: dict[str, Any] | None = None

    if args.max_batches:
        epoch = int(state["completed_epochs"]) + 1
        train_metrics = run_epoch(model, train_loader, device, optimizer, args.max_batches)
        validation_metrics = run_epoch(model, validation_loader, device, None, args.max_batches)
        last_record = {"epoch": epoch, "full_epoch": False, "max_batches": args.max_batches, "train": train_metrics, "validation": validation_metrics}
        update_training_state(state, last_record, full_epoch=False, optimizer_steps=train_metrics["batches"], best_path=best_path)
        save_checkpoint(last_path, model, optimizer, state, actual_config, last_record, audit["input_sha256"], code_commit)
        print(json.dumps(last_record), flush=True)
    else:
        while int(state["completed_epochs"]) < args.epochs and int(state["stale"]) < args.patience:
            epoch = int(state["completed_epochs"]) + 1
            train_metrics = run_epoch(model, train_loader, device, optimizer)
            validation_metrics = run_epoch(model, validation_loader, device)
            last_record = {"epoch": epoch, "full_epoch": True, "max_batches": 0, "train": train_metrics, "validation": validation_metrics}
            improved = update_training_state(state, last_record, full_epoch=True, optimizer_steps=train_metrics["batches"], best_path=best_path)
            full_epochs_this_run += 1
            if improved:
                save_checkpoint(best_path, model, optimizer, state, actual_config, last_record, audit["input_sha256"], code_commit)
            save_checkpoint(last_path, model, optimizer, state, actual_config, last_record, audit["input_sha256"], code_commit)
            print(json.dumps(last_record), flush=True)
    if last_record is None:
        raise RuntimeError("no training epoch or partial epoch was executed")

    chosen_path = Path(state["best_checkpoint_path"]) if state["best_checkpoint_path"] else last_path
    chosen = torch.load(chosen_path, map_location=device, weights_only=False)
    model.load_state_dict(chosen["model"])
    final_limit = args.max_batches if args.max_batches else 0
    final_loss = run_epoch(model, validation_loader, device, max_batches=final_limit)
    decoding = evaluate_decoding(model, validation_loader, device, max_batches=final_limit)
    if decoding["decode_legal_rate"] != 1.0 or decoding["invalid_actions"] != 0:
        raise RuntimeError("validation decode legality gate failed")

    prior_rows = read_run_rows(Path(args.runs_output))
    prior_formal_seeds = {int(row["seed"]) for row in prior_rows if row.get("status") in {"partial_formal", "completed_formal"} and row.get("seed", "").isdigit()}
    current_config_matches = not args.max_batches and args.seed in planned_seeds and args.epochs == planned_epochs and args.split_seed == int(planned_config["split"]["seed"])
    achieved_seeds = prior_formal_seeds | ({args.seed} if current_config_matches else set())
    missing_seeds = sorted(set(planned_seeds) - achieved_seeds)
    if args.max_batches:
        status = "smoke"
    else:
        status = "completed_formal" if not missing_seeds and current_config_matches else "partial_formal"
    actual_config.update({
        "full_epochs_this_run": full_epochs_this_run, "completed_epochs": state["completed_epochs"],
        "optimizer_steps": state["optimizer_steps"], "early_stopped": state["stale"] >= args.patience,
        "missing_seeds": missing_seeds,
    })
    runtime = time.perf_counter() - start
    checkpoint_hash = checkpoint_sha256(chosen_path)
    result = {
        "experiment_id": args.experiment_id, "status": status, "partial_run": bool(args.max_batches),
        "planned_config": planned_config, "actual_config": actual_config,
        "provenance": {"git_sha": code_commit, "dirty": bool(initial_dirty), "input_sha256": audit["input_sha256"]},
        "dataset_audit": audit,
        "split": {key: manifest[key] for key in ("split_seed", "validation_fraction", "train_episodes", "validation_episodes", "episode_overlap", "rows", "policy_rows", "value_rows")},
        "train_rows": len(train), "validation_rows": len(validation),
        "epochs_completed": state["completed_epochs"], "full_epochs_this_run": full_epochs_this_run,
        "best_epoch": state["best_epoch"], "best_validation_loss": state["best_loss"],
        "training_state": state, "validation_loss": final_loss, "validation": decoding,
        "checkpoint": {"path": str(chosen_path), "sha256": checkpoint_hash, "bytes": chosen_path.stat().st_size},
        "runtime_seconds": runtime, "peak_ram_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0,
        "missing_seeds": missing_seeds, "failures": [],
    }
    write_json(Path(args.metrics_output), result)
    if not args.max_batches:
        append_run_csv(Path(args.runs_output), {
            "experiment_id": args.experiment_id, "seed": args.seed, "status": status,
            "epochs_completed": state["completed_epochs"], "best_epoch": state["best_epoch"], "best_validation_loss": state["best_loss"],
            "sequence_exact_match": decoding["sequence_exact_match"], "set_exact_match": decoding["set_exact_match"],
            "empty_action_accuracy": decoding["empty_action_accuracy"], "multi_select_accuracy": decoding["multi_select_accuracy"],
            "decode_legal_rate": decoding["decode_legal_rate"], "invalid_actions": decoding["invalid_actions"],
            "unsupported_rows": audit["unsupported_rows"], "skipped_rows": audit["skipped_rows"],
            "checkpoint": chosen_path, "checkpoint_sha256": checkpoint_hash, "runtime_seconds": runtime,
            "peak_ram_mb": result["peak_ram_mb"], "peak_vram_mb": result["peak_vram_mb"], "git_sha": code_commit,
            "dirty": bool(initial_dirty), "planned_seeds": json.dumps(planned_seeds), "missing_seeds": json.dumps(missing_seeds),
            "planned_epochs": planned_epochs, "actual_epochs": state["completed_epochs"],
        })
    print(json.dumps({key: result[key] for key in ("status", "epochs_completed", "best_epoch", "missing_seeds", "runtime_seconds", "checkpoint")}), flush=True)


from rl.bc import TrajectoryDataset, split_by_episode  # noqa: E402,F401


if __name__ == "__main__":
    main()
