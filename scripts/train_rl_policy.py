from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
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
    sha256_file,
)
from rl.features import attack_metadata_table, card_metadata_table
from rl.model import MaskedPointerActorCritic, StructuredMaskedPointerActorCritic


ARCHITECTURE = "stateless_masked_autoregressive_candidate_pointer_with_stop"
HISTORY_ARCHITECTURE = "causal_gru_history_masked_autoregressive_candidate_pointer_with_stop"
STRUCTURED_ARCHITECTURE = "structured_card_attack_deepsets_deck_masked_pointer_with_stop"
HISTORY_GROUP_BY = ["episode_id", "player"]
HISTORY_ORDER_BY = "action_step"
HISTORY_TOKEN = "prior pre-action state plus that prior selected-option summary"


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


def actual_fingerprint_payload(
    *, code_commit: str, input_sha256: str, split_seed: int, validation_fraction: float,
    epochs: int, batch_size: int, learning_rate: float, hidden_dim: int, patience: int,
    value_loss_weight: float, gradient_clip_norm: float, architecture: str = ARCHITECTURE,
    history_length: int = 0, experiment_id: str | None = None,
    structured: dict[str, Any] | None = None,
    initialization: str = "random",
) -> dict[str, Any]:
    history_enabled = architecture == HISTORY_ARCHITECTURE
    payload = {
        "code_commit": code_commit,
        "input_sha256": input_sha256,
        "split": {"seed": split_seed, "validation_fraction": validation_fraction},
        "training": {
            "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
            "hidden_dim": hidden_dim, "patience": patience, "value_loss_weight": value_loss_weight,
            "gradient_clip_norm": gradient_clip_norm,
        },
        "architecture": architecture,
        "history": {
            "enabled": history_enabled,
            "encoder": "gru" if history_enabled else "none",
            "max_length": history_length if history_enabled else 0,
        },
        "initialization": initialization,
        "offline_rl": False,
    }
    if history_enabled:
        payload["history"].update({
            "group_by": HISTORY_GROUP_BY,
            "order_by": HISTORY_ORDER_BY,
            "token": HISTORY_TOKEN,
        })
    if structured is not None:
        payload["structured"] = structured
    if experiment_id is not None:
        payload["experiment_id"] = experiment_id
    return payload


def experiment_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def current_config_matches(planned: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, list[str]]:
    checks = {
        "input_sha256": actual["input_sha256"] == planned["input_sha256"],
        "split_seed": actual["split"]["seed"] == planned["split"]["seed"],
        "validation_fraction": math.isclose(actual["split"]["validation_fraction"], planned["split"]["validation_fraction"], rel_tol=0.0, abs_tol=1e-12),
        "epochs": actual["training"]["epochs"] == planned["training"]["epochs"],
        "batch_size": actual["training"]["batch_size"] == planned["training"]["batch_size"],
        "learning_rate": math.isclose(actual["training"]["learning_rate"], planned["training"]["learning_rate"], rel_tol=0.0, abs_tol=1e-15),
        "hidden_dim": actual["training"]["hidden_dim"] == planned["training"]["hidden_dim"],
        "patience": actual["training"]["patience"] == planned["training"]["early_stopping_patience"],
        "value_loss_weight": math.isclose(actual["training"]["value_loss_weight"], planned["training"]["value_loss_weight"], rel_tol=0.0, abs_tol=1e-15),
        "gradient_clip_norm": math.isclose(actual["training"]["gradient_clip_norm"], planned["training"]["gradient_clip_norm"], rel_tol=0.0, abs_tol=1e-15),
        "architecture": actual["architecture"] == planned["architecture"],
    }
    if "history" in planned:
        checks.update({
            "history_enabled": actual["history"]["enabled"] == planned["history"]["enabled"],
            "history_encoder": actual["history"]["encoder"] == planned["history"]["encoder"],
            "history_max_length": actual["history"]["max_length"] == planned["history"]["max_length"],
        })
        for field in ("group_by", "order_by", "token"):
            if field in planned["history"]:
                checks[f"history_{field}"] = actual["history"].get(field) == planned["history"][field]
    if "structured" in planned:
        for field, expected in planned["structured"].items():
            checks[f"structured_{field}"] = actual.get("structured", {}).get(field) == expected
    if "experiment_id" in planned:
        checks["experiment_id"] = actual.get("experiment_id") == planned["experiment_id"]
    if "random_initialization" in planned:
        checks["random_initialization"] = (
            actual["initialization"] == "random"
            if planned["random_initialization"]
            else actual["initialization"] in {"resume", "warm_start"}
        )
    if "initialization" in planned:
        checks["initialization"] = actual["initialization"] == planned["initialization"]
    if "offline_rl" in planned:
        checks["offline_rl"] = actual["offline_rl"] == planned["offline_rl"]
    mismatches = [name for name, matched in checks.items() if not matched]
    return not mismatches, mismatches


def achieved_seeds_for_fingerprint(rows: list[dict[str, str]], fingerprint: str) -> set[int]:
    return {
        int(row["seed"])
        for row in rows
        if row.get("experiment_fingerprint") == fingerprint
        and row.get("config_matched", "").lower() == "true"
        and row.get("seed", "").isdigit()
    }


def validate_resume_compatibility(checkpoint: dict[str, Any], *, code_commit: str, input_sha256: str, fingerprint: str) -> None:
    mismatches = []
    if checkpoint.get("git_sha") != code_commit:
        mismatches.append("git_sha")
    if checkpoint.get("input_sha256") != input_sha256:
        mismatches.append("input_sha256")
    if checkpoint.get("experiment_fingerprint") != fingerprint:
        mismatches.append("experiment_fingerprint")
    if mismatches:
        raise ValueError(f"resume checkpoint mismatch: {', '.join(mismatches)}")


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


def save_checkpoint(path: Path, model: MaskedPointerActorCritic, optimizer: torch.optim.Optimizer, training_state: dict[str, Any], config: dict[str, Any], metrics: dict[str, Any], input_sha256: str, code_commit: str, fingerprint: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "training_state": copy.deepcopy(training_state), "config": config, "metrics": metrics,
        "input_sha256": input_sha256, "hidden_dim": model.hidden_dim, "git_sha": code_commit,
        "experiment_fingerprint": fingerprint,
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
    "experiment_fingerprint", "config_matched", "config_mismatches",
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
    parser = argparse.ArgumentParser(description="Masked behavior cloning with optional structured card/deck inputs")
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
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--architecture", choices=(ARCHITECTURE, HISTORY_ARCHITECTURE, STRUCTURED_ARCHITECTURE),
        default=ARCHITECTURE,
    )
    parser.add_argument("--history-length", type=int, default=0)
    parser.add_argument("--deck-map", default="data/processed/replay_decks_2026-08-05.jsonl.gz")
    parser.add_argument("--card-database", default="data/reference/official_cards.json")
    parser.add_argument("--attack-database", default="data/reference/official_attacks.json")
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-dir", default="checkpoints/rl_bc_001")
    parser.add_argument("--metrics-output", default="results/rl_bc_001_metrics.json")
    parser.add_argument("--split-output", default="results/rl_bc_001_split.json")
    parser.add_argument("--runs-output", default="results/rl_bc_001_runs.csv")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--initialize-from",
        default=None,
        help="Warm-start model weights from a checkpoint while resetting optimizer and epoch state",
    )
    parser.add_argument("--max-batches", type=int, default=0, help="Smoke-only limit; never completes an epoch")
    args = parser.parse_args()

    if args.resume and args.initialize_from:
        raise ValueError("--resume and --initialize-from are mutually exclusive")

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

    history_enabled = args.architecture == HISTORY_ARCHITECTURE
    structured_enabled = args.architecture == STRUCTURED_ARCHITECTURE
    if history_enabled and args.history_length <= 0:
        raise ValueError("history architecture requires --history-length > 0")
    if not history_enabled and args.history_length != 0:
        raise ValueError("stateless architecture requires --history-length 0")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be in [0, 1]")
    rows, audit = load_replay_dataset(
        Path(args.input), history_length=args.history_length,
        deck_map_path=Path(args.deck_map) if structured_enabled else None,
        structured=structured_enabled,
    )
    structured_assets = None
    if structured_enabled:
        structured_assets = {
            "card_attack_embeddings": True,
            "entity_encoder": "deepsets_masked_mean_max",
            "deck_conditioning": "acting_player_submitted_deck_masked_mean",
            "deck_map_sha256": audit["deck_map"]["sha256"],
            "card_database_sha256": sha256_file(Path(args.card_database)),
            "attack_database_sha256": sha256_file(Path(args.attack_database)),
            "confidence_threshold": args.confidence_threshold,
        }
    initialization = "resume" if args.resume else ("warm_start" if args.initialize_from else "random")
    fingerprint_payload = actual_fingerprint_payload(
        code_commit=code_commit, input_sha256=audit["input_sha256"], split_seed=args.split_seed,
        validation_fraction=args.validation_fraction, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, hidden_dim=args.hidden_dim, patience=args.patience,
        value_loss_weight=args.value_loss_weight, gradient_clip_norm=args.gradient_clip_norm,
        architecture=args.architecture, history_length=args.history_length,
        experiment_id=args.experiment_id,
        structured=structured_assets,
        initialization=initialization,
    )
    fingerprint = experiment_fingerprint(fingerprint_payload)
    config_matched, config_mismatches = current_config_matches(planned_config, fingerprint_payload)
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
    if structured_enabled:
        model = StructuredMaskedPointerActorCritic(
            args.hidden_dim,
            card_metadata=torch.tensor(card_metadata_table(Path(args.card_database))),
            attack_metadata=torch.tensor(attack_metadata_table(Path(args.attack_database))),
        ).to(device)
    else:
        model = MaskedPointerActorCritic(args.hidden_dim, history_encoder=history_enabled).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    state = new_training_state()
    warm_start: dict[str, Any] | None = None
    if args.initialize_from:
        source_path = Path(args.initialize_from).resolve()
        checkpoint = torch.load(source_path, map_location=device, weights_only=False)
        source_hidden_dim = int(checkpoint.get("hidden_dim", -1))
        if source_hidden_dim != args.hidden_dim:
            raise ValueError(
                f"warm-start hidden_dim mismatch: checkpoint={source_hidden_dim}, requested={args.hidden_dim}"
            )
        model.load_state_dict(checkpoint["model"], strict=True)
        warm_start = {
            "path": str(source_path),
            "sha256": checkpoint_sha256(source_path),
            "git_sha": checkpoint.get("git_sha"),
            "experiment_fingerprint": checkpoint.get("experiment_fingerprint"),
        }
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        validate_resume_compatibility(
            checkpoint, code_commit=code_commit, input_sha256=audit["input_sha256"], fingerprint=fingerprint
        )
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        state = restore_training_state(checkpoint)

    actual_config = vars(args).copy()
    actual_config.update({
        "code_commit": code_commit, "dirty_at_start": bool(initial_dirty), "input_sha256": audit["input_sha256"],
        "device_resolved": str(device), "stateless": not history_enabled, "architecture": args.architecture,
        "history_encoder": history_enabled, "history_length": args.history_length,
        "structured": structured_assets,
        "initialization": initialization, "warm_start": warm_start,
        "policy_rows": "winner_only", "value_rows": "both_players",
        "value_loss_weight": args.value_loss_weight, "gradient_clip_norm": args.gradient_clip_norm,
        "experiment_fingerprint": fingerprint, "config_matched": config_matched,
        "config_mismatches": config_mismatches,
    })
    checkpoint_dir = Path(args.checkpoint_dir)
    best_path = checkpoint_dir / f"seed_{args.seed}_best.pt"
    last_path = checkpoint_dir / f"seed_{args.seed}_last.pt"
    full_epochs_this_run = 0
    last_record: dict[str, Any] | None = None

    if args.max_batches:
        epoch = int(state["completed_epochs"]) + 1
        train_metrics = run_epoch(model, train_loader, device, optimizer, args.max_batches, args.value_loss_weight, args.gradient_clip_norm)
        validation_metrics = run_epoch(model, validation_loader, device, None, args.max_batches, args.value_loss_weight, args.gradient_clip_norm)
        last_record = {"epoch": epoch, "full_epoch": False, "max_batches": args.max_batches, "train": train_metrics, "validation": validation_metrics}
        update_training_state(state, last_record, full_epoch=False, optimizer_steps=train_metrics["batches"], best_path=best_path)
        save_checkpoint(last_path, model, optimizer, state, actual_config, last_record, audit["input_sha256"], code_commit, fingerprint)
        print(json.dumps(last_record), flush=True)
    else:
        while int(state["completed_epochs"]) < args.epochs and int(state["stale"]) < args.patience:
            epoch = int(state["completed_epochs"]) + 1
            train_metrics = run_epoch(model, train_loader, device, optimizer, value_loss_weight=args.value_loss_weight, gradient_clip_norm=args.gradient_clip_norm)
            validation_metrics = run_epoch(model, validation_loader, device, value_loss_weight=args.value_loss_weight, gradient_clip_norm=args.gradient_clip_norm)
            last_record = {"epoch": epoch, "full_epoch": True, "max_batches": 0, "train": train_metrics, "validation": validation_metrics}
            improved = update_training_state(state, last_record, full_epoch=True, optimizer_steps=train_metrics["batches"], best_path=best_path)
            full_epochs_this_run += 1
            if improved:
                save_checkpoint(best_path, model, optimizer, state, actual_config, last_record, audit["input_sha256"], code_commit, fingerprint)
            save_checkpoint(last_path, model, optimizer, state, actual_config, last_record, audit["input_sha256"], code_commit, fingerprint)
            print(json.dumps(last_record), flush=True)
    if last_record is None:
        raise RuntimeError("no training epoch or partial epoch was executed")

    chosen_path = Path(state["best_checkpoint_path"]) if state["best_checkpoint_path"] else last_path
    chosen = torch.load(chosen_path, map_location=device, weights_only=False)
    model.load_state_dict(chosen["model"])
    final_limit = args.max_batches if args.max_batches else 0
    final_loss = run_epoch(model, validation_loader, device, max_batches=final_limit, value_loss_weight=args.value_loss_weight, gradient_clip_norm=args.gradient_clip_norm)
    decoding = evaluate_decoding(model, validation_loader, device, max_batches=final_limit)
    if decoding["decode_legal_rate"] != 1.0 or decoding["invalid_actions"] != 0:
        raise RuntimeError("validation decode legality gate failed")

    prior_rows = read_run_rows(Path(args.runs_output))
    prior_formal_seeds = achieved_seeds_for_fingerprint(prior_rows, fingerprint)
    current_run_matches = not args.max_batches and config_matched and args.seed in planned_seeds
    achieved_seeds = prior_formal_seeds | ({args.seed} if current_run_matches else set())
    missing_seeds = sorted(set(planned_seeds) - achieved_seeds)
    if args.max_batches:
        status = "smoke"
    else:
        status = "exploratory_config_mismatch" if not config_matched else ("completed_formal" if not missing_seeds else "partial_formal")
    actual_config.update({
        "full_epochs_this_run": full_epochs_this_run, "completed_epochs": state["completed_epochs"],
        "optimizer_steps": state["optimizer_steps"], "early_stopped": state["stale"] >= args.patience,
        "missing_seeds": missing_seeds, "achieved_seeds": sorted(achieved_seeds),
    })
    runtime = time.perf_counter() - start
    checkpoint_hash = checkpoint_sha256(chosen_path)
    result = {
        "experiment_id": args.experiment_id, "status": status, "partial_run": bool(args.max_batches),
        "planned_config": planned_config, "actual_config": actual_config,
        "provenance": {"git_sha": code_commit, "dirty": bool(initial_dirty), "input_sha256": audit["input_sha256"], "experiment_fingerprint": fingerprint},
        "experiment_fingerprint_payload": fingerprint_payload,
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
            "experiment_fingerprint": fingerprint, "config_matched": config_matched,
            "config_mismatches": json.dumps(config_mismatches),
        })
    print(json.dumps({key: result[key] for key in ("status", "epochs_completed", "best_epoch", "missing_seeds", "runtime_seconds", "checkpoint")}), flush=True)


from rl.bc import TrajectoryDataset, split_by_episode  # noqa: E402,F401


if __name__ == "__main__":
    main()
