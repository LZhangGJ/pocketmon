from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.bc import (
    TrajectoryDataset,
    build_split_manifest,
    evaluate_decoding,
    load_replay_dataset,
    make_loader,
    run_epoch,
    sha256_file,
)
from rl.features import attack_metadata_table, card_metadata_table
from rl.temporal_model import (
    TEMPORAL_ARCHITECTURE,
    TEMPORAL_STATE_PREFIXES,
    StructuredTemporalTransformerActorCritic,
    load_structured_warm_start,
)
from scripts.train_rl_policy import (
    assert_formal_worktree_clean,
    checkpoint_sha256,
    choose_device,
    git_sha,
    load_torch_checkpoint,
    peak_ram_mb,
    seed_everything,
    write_json,
)


STRUCTURED_ARCHITECTURE = (
    "structured_card_attack_deepsets_deck_masked_pointer_with_stop"
)
HISTORY_GROUP_BY = ["episode_id", "player"]
HISTORY_ORDER_BY = "action_step"
HISTORY_TOKEN = "prior pre-action state plus that prior selected-option summary"


def _same(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return math.isclose(
                float(expected), float(actual), rel_tol=0.0, abs_tol=1e-12
            )
        except (TypeError, ValueError):
            return False
    return expected == actual


def subset_mismatches(
    expected: dict[str, Any],
    actual: dict[str, Any],
    prefix: str = "",
) -> list[str]:
    """Compare all planned fields while allowing extra runtime metadata."""

    mismatches: list[str] = []
    for key, value in expected.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in actual:
            mismatches.append(f"{path}:missing")
            continue
        observed = actual[key]
        if isinstance(value, dict):
            if not isinstance(observed, dict):
                mismatches.append(f"{path}:type")
            else:
                mismatches.extend(subset_mismatches(value, observed, path))
        elif not _same(value, observed):
            mismatches.append(f"{path}:expected={value!r}:actual={observed!r}")
    return mismatches


def experiment_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def temporal_parameters(
    model: StructuredTemporalTransformerActorCritic,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    temporal: list[torch.nn.Parameter] = []
    base: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if name.startswith(TEMPORAL_STATE_PREFIXES):
            temporal.append(parameter)
        else:
            base.append(parameter)
    if not temporal or not base:
        raise RuntimeError("failed to partition temporal and base parameters")
    return temporal, base


def save_checkpoint(
    path: Path,
    *,
    model: StructuredTemporalTransformerActorCritic,
    optimizer: torch.optim.Optimizer,
    training_state: dict[str, Any],
    config: dict[str, Any],
    metrics: dict[str, Any],
    input_sha256: str,
    code_commit: str,
    fingerprint: str,
    warm_start: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "kind": "rl_bc_004_structured_temporal_transformer",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_state": training_state,
            "config": config,
            "metrics": metrics,
            "input_sha256": input_sha256,
            "hidden_dim": model.hidden_dim,
            "git_sha": code_commit,
            "experiment_fingerprint": fingerprint,
            "warm_start": warm_start,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RL-BC-004 fixed-deck structured BC with an eight-decision "
            "causal Transformer history"
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--deck-map", type=Path, required=True)
    parser.add_argument("--target-deck", type=Path, required=True)
    parser.add_argument("--planned-config", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=20260720)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--base-learning-rate", type=float, default=1e-4)
    parser.add_argument("--temporal-learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--value-loss-weight", type=float, default=0.25)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ffn-dim", type=int, default=768)
    parser.add_argument("--transformer-dropout", type=float, default=0.10)
    parser.add_argument(
        "--card-database",
        type=Path,
        default=Path("data/reference/official_cards.json"),
    )
    parser.add_argument(
        "--attack-database",
        type=Path,
        default=Path("data/reference/official_attacks.json"),
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    if args.history_length != 8:
        raise ValueError("RL-BC-004 MVP is pre-registered with history_length=8")
    if args.epochs != 12:
        raise ValueError("RL-BC-004 MVP is pre-registered for 12 epochs")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be in [0, 1]")
    for path in (
        args.input,
        args.deck_map,
        args.target_deck,
        args.planned_config,
        args.initialize_from,
        args.card_database,
        args.attack_database,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.metrics_output.exists() or args.split_output.exists():
        raise FileExistsError("refusing to overwrite experiment evidence")

    started = time.perf_counter()
    dirty = assert_formal_worktree_clean(args.max_batches)
    code_commit = git_sha()
    seed_everything(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
        torch.cuda.reset_peak_memory_stats()

    rows, audit = load_replay_dataset(
        args.input,
        history_length=args.history_length,
        deck_map_path=args.deck_map,
        structured=True,
    )
    if not audit.get("structured"):
        raise AssertionError("structured replay features were not enabled")
    if not audit.get("history", {}).get("enabled"):
        raise AssertionError("causal history was not enabled")
    if audit["history"].get("max_length") != args.history_length:
        raise AssertionError("history audit length mismatch")
    if audit["history"].get("current_or_future_steps_used") != 0:
        raise AssertionError("non-causal history detected")
    if audit.get("forbidden_model_fields_used"):
        raise AssertionError("forbidden replay fields reached the model")

    manifest, train_ids, validation_ids = build_split_manifest(
        rows,
        args.validation_fraction,
        args.split_seed,
    )
    if train_ids & validation_ids or manifest["episode_overlap"]:
        raise AssertionError("episode leakage detected")
    manifest.update(
        {
            "input_sha256": audit["input_sha256"],
            "deck_map_sha256": audit["deck_map"]["sha256"],
            "target_deck_sha256": sha256_file(args.target_deck),
            "code_commit": code_commit,
            "dirty_at_start": bool(dirty),
        }
    )
    write_json(args.split_output, manifest)

    train_rows = [row for row in rows if row["episode_id"] in train_ids]
    validation_rows = [row for row in rows if row["episode_id"] in validation_ids]
    if not train_rows or not validation_rows:
        raise ValueError("empty train or validation split")
    train_loader = make_loader(
        TrajectoryDataset(train_rows),
        args.batch_size,
        True,
        args.seed,
    )
    validation_loader = make_loader(
        TrajectoryDataset(validation_rows),
        args.batch_size,
        False,
        args.seed,
    )

    model = StructuredTemporalTransformerActorCritic(
        hidden_dim=args.hidden_dim,
        card_metadata=torch.tensor(
            card_metadata_table(args.card_database),
            dtype=torch.float32,
        ),
        attack_metadata=torch.tensor(
            attack_metadata_table(args.attack_database),
            dtype=torch.float32,
        ),
        history_length=args.history_length,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_ffn_dim=args.transformer_ffn_dim,
        transformer_dropout=args.transformer_dropout,
    ).to(device)

    source = load_torch_checkpoint(args.initialize_from, device)
    if int(source.get("hidden_dim", -1)) != args.hidden_dim:
        raise ValueError("warm-start hidden_dim mismatch")
    source_architecture = (
        source.get("config", {}).get("architecture", STRUCTURED_ARCHITECTURE)
    )
    if source_architecture == TEMPORAL_ARCHITECTURE:
        model.load_state_dict(source["model"], strict=True)
        warm_start_load = {"missing_keys": [], "unexpected_keys": []}
    elif source_architecture == STRUCTURED_ARCHITECTURE:
        warm_start_load = load_structured_warm_start(model, source["model"])
    else:
        raise ValueError(
            f"unsupported warm-start architecture: {source_architecture}"
        )
    warm_start = {
        "path": str(args.initialize_from.resolve()),
        "sha256": checkpoint_sha256(args.initialize_from),
        "source_architecture": source_architecture,
        "source_git_sha": source.get("git_sha"),
        "load": warm_start_load,
    }

    temporal, base = temporal_parameters(model)
    optimizer = torch.optim.Adam(
        [
            {"params": base, "lr": args.base_learning_rate, "name": "base"},
            {
                "params": temporal,
                "lr": args.temporal_learning_rate,
                "name": "temporal",
            },
        ]
    )

    plan_actual = {
        "experiment_id": args.experiment_id,
        "arm": "fixed-deck-structured-transformer8-warm-start",
        "architecture": TEMPORAL_ARCHITECTURE,
        "input": str(args.input.resolve()),
        "input_sha256": audit["input_sha256"],
        "dataset": {
            "deck_map_sha256": audit["deck_map"]["sha256"],
            "target_deck_sha256": sha256_file(args.target_deck),
        },
        "split": {
            "kind": "episode_id",
            "seed": args.split_seed,
            "train_fraction": 1.0 - args.validation_fraction,
            "validation_fraction": args.validation_fraction,
        },
        "training": {
            "formal_seeds": [args.seed],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "base_learning_rate": args.base_learning_rate,
            "temporal_learning_rate": args.temporal_learning_rate,
            "hidden_dim": args.hidden_dim,
            "early_stopping_patience": args.patience,
            "value_loss_weight": args.value_loss_weight,
            "gradient_clip_norm": args.gradient_clip_norm,
        },
        "history": {
            "enabled": True,
            "encoder": "transformer",
            "max_length": args.history_length,
            "group_by": HISTORY_GROUP_BY,
            "order_by": HISTORY_ORDER_BY,
            "token": HISTORY_TOKEN,
        },
        "transformer": {
            "layers": args.transformer_layers,
            "heads": args.transformer_heads,
            "ffn_dim": args.transformer_ffn_dim,
            "dropout": args.transformer_dropout,
        },
        "structured": {
            "card_attack_embeddings": True,
            "entity_encoder": "deepsets_masked_mean_max",
            "deck_conditioning": "acting_player_submitted_deck_masked_mean",
            "card_database_sha256": sha256_file(args.card_database),
            "attack_database_sha256": sha256_file(args.attack_database),
            "confidence_threshold": args.confidence_threshold,
        },
        "warm_start": {
            "checkpoint_sha256": warm_start["sha256"],
            "source_architecture": source_architecture,
        },
        "policy_loss_rows": "policy_weight == 1 only",
        "value_loss_rows": "value_weight == 1 for both players",
        "initialization": "warm_start",
        "offline_rl": False,
    }
    planned = json.loads(args.planned_config.read_text(encoding="utf-8"))
    mismatches = subset_mismatches(planned, plan_actual)
    if mismatches and not args.max_batches:
        raise ValueError(
            "formal configuration does not match the planned config:\n"
            + "\n".join(mismatches)
        )

    fingerprint_payload = {
        **plan_actual,
        "code_commit": code_commit,
    }
    fingerprint = experiment_fingerprint(fingerprint_payload)
    checkpoint_config = {
        "experiment_id": args.experiment_id,
        "architecture": TEMPORAL_ARCHITECTURE,
        "history_length": args.history_length,
        "transformer_layers": args.transformer_layers,
        "transformer_heads": args.transformer_heads,
        "transformer_ffn_dim": args.transformer_ffn_dim,
        "transformer_dropout": args.transformer_dropout,
        "confidence_threshold": args.confidence_threshold,
        "structured": True,
        "target_deck_sha256": sha256_file(args.target_deck),
        "deck_map_sha256": audit["deck_map"]["sha256"],
        "planned_config": planned,
        "config_mismatches": mismatches,
    }

    state: dict[str, Any] = {
        "best_loss": float("inf"),
        "best_epoch": 0,
        "stale": 0,
        "history": [],
        "completed_epochs": 0,
        "best_checkpoint_path": None,
    }
    best_path = args.checkpoint_dir / f"seed_{args.seed}_best.pt"
    last_path = args.checkpoint_dir / f"seed_{args.seed}_last.pt"
    last_record: dict[str, Any] | None = None

    if args.max_batches:
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            max_batches=args.max_batches,
            value_loss_weight=args.value_loss_weight,
            gradient_clip_norm=args.gradient_clip_norm,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            max_batches=args.max_batches,
            value_loss_weight=args.value_loss_weight,
            gradient_clip_norm=args.gradient_clip_norm,
        )
        last_record = {
            "epoch": 1,
            "full_epoch": False,
            "max_batches": args.max_batches,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        state["best_loss"] = float(validation_metrics["loss"])
        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            training_state=state,
            config=checkpoint_config,
            metrics=last_record,
            input_sha256=audit["input_sha256"],
            code_commit=code_commit,
            fingerprint=fingerprint,
            warm_start=warm_start,
        )
        chosen_path = last_path
    else:
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                value_loss_weight=args.value_loss_weight,
                gradient_clip_norm=args.gradient_clip_norm,
            )
            validation_metrics = run_epoch(
                model,
                validation_loader,
                device,
                value_loss_weight=args.value_loss_weight,
                gradient_clip_norm=args.gradient_clip_norm,
            )
            last_record = {
                "epoch": epoch,
                "full_epoch": True,
                "max_batches": 0,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            state["history"].append(last_record)
            state["completed_epochs"] = epoch
            validation_loss = float(validation_metrics["loss"])
            if validation_loss < float(state["best_loss"]):
                state["best_loss"] = validation_loss
                state["best_epoch"] = epoch
                state["stale"] = 0
                state["best_checkpoint_path"] = str(best_path)
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    training_state=state,
                    config=checkpoint_config,
                    metrics=last_record,
                    input_sha256=audit["input_sha256"],
                    code_commit=code_commit,
                    fingerprint=fingerprint,
                    warm_start=warm_start,
                )
            else:
                state["stale"] += 1
            save_checkpoint(
                last_path,
                model=model,
                optimizer=optimizer,
                training_state=state,
                config=checkpoint_config,
                metrics=last_record,
                input_sha256=audit["input_sha256"],
                code_commit=code_commit,
                fingerprint=fingerprint,
                warm_start=warm_start,
            )
            print(json.dumps(last_record), flush=True)
            if int(state["stale"]) >= args.patience:
                break
        chosen_path = (
            Path(state["best_checkpoint_path"])
            if state["best_checkpoint_path"]
            else last_path
        )

    if last_record is None:
        raise RuntimeError("no training work was executed")
    chosen = load_torch_checkpoint(chosen_path, device)
    model.load_state_dict(chosen["model"], strict=True)
    final_limit = args.max_batches if args.max_batches else 0
    final_loss = run_epoch(
        model,
        validation_loader,
        device,
        max_batches=final_limit,
        value_loss_weight=args.value_loss_weight,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    decoding = evaluate_decoding(
        model,
        validation_loader,
        device,
        max_batches=final_limit,
    )
    if decoding["decode_legal_rate"] != 1.0 or decoding["invalid_actions"]:
        raise RuntimeError("validation decode legality gate failed")

    result = {
        "experiment_id": args.experiment_id,
        "status": "smoke" if args.max_batches else "completed",
        "partial_run": bool(args.max_batches),
        "planned_config": planned,
        "actual_plan": plan_actual,
        "config_mismatches": mismatches,
        "provenance": {
            "git_sha": code_commit,
            "dirty": bool(dirty),
            "input_sha256": audit["input_sha256"],
            "experiment_fingerprint": fingerprint,
        },
        "dataset_audit": audit,
        "split": {
            key: manifest[key]
            for key in (
                "split_seed",
                "validation_fraction",
                "train_episodes",
                "validation_episodes",
                "episode_overlap",
                "rows",
                "policy_rows",
                "value_rows",
            )
        },
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "training_state": state,
        "validation_loss": final_loss,
        "validation": decoding,
        "warm_start": warm_start,
        "checkpoint": {
            "path": str(chosen_path.resolve()),
            "sha256": checkpoint_sha256(chosen_path),
            "bytes": chosen_path.stat().st_size,
        },
        "runtime_seconds": time.perf_counter() - started,
        "peak_ram_mb": peak_ram_mb(),
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if device.type == "cuda"
            else 0.0
        ),
        "failures": [],
    }
    write_json(args.metrics_output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "best_epoch": state["best_epoch"],
                "checkpoint": result["checkpoint"],
                "validation": {
                    "sequence_exact_match": decoding["sequence_exact_match"],
                    "set_exact_match": decoding["set_exact_match"],
                    "decode_legal_rate": decoding["decode_legal_rate"],
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
