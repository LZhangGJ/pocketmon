from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import Experiment7Error, read_json, sha256_file, utc_now, write_json


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def setup_vendor(reference_root: Path) -> dict[str, Any]:
    training = reference_root / "training"
    pipeline = reference_root / "data_pipeline"
    for path in (training, pipeline):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from deck_identity_model import DeckIdentityModelConfig, PTCGDeckIdentityTransformerPolicy
    from train import MODULE_WEIGHTS, decisions_for_episodes, episode_order, load_module_by_episode
    from train_multideck_identity import (
        IdentityBundle,
        evaluate_identity,
        save_checkpoint,
        train_balanced_multideck_epoch,
        train_single_epoch,
    )

    return locals()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise Experiment7Error("CUDA requested but torch.cuda.is_available() is false")
    return device


def source_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    required = {"referenceRoot", "classMap", "catalog", "pretrain", "currentSources", "engineCatalog"}
    missing = required - set(payload)
    if missing:
        raise Experiment7Error(f"training source manifest missing: {sorted(missing)}")
    return payload


def load_bundle(vendor: dict[str, Any], name: str, row: dict[str, Any]):
    IdentityBundle = vendor["IdentityBundle"]
    return IdentityBundle.load(
        name,
        Path(row["features"]),
        Path(row["tokenCache"]),
        Path(row["sequenceCache"]),
        Path(row["identityCache"]),
    )


def current_splits(vendor: dict[str, Any], manifest: dict[str, Any]):
    episode_order = vendor["episode_order"]
    decisions_for_episodes = vendor["decisions_for_episodes"]
    bundles = []
    fit_sources = []
    split_data = {}
    for row in manifest["currentSources"]:
        bundle = load_bundle(vendor, row["name"], row)
        base = bundle.sequence.base
        meaningful = base.nontrivial_mask()
        holdout = np.flatnonzero(base.data["validation"] == 1)
        train = np.flatnonzero((base.data["validation"] == 0) & meaningful)
        train_episodes = episode_order(base, train)
        calibration_count = int(row["calibrationEpisodes"])
        if calibration_count <= 0 or calibration_count >= len(train_episodes):
            raise Experiment7Error(
                f"{row['name']}: invalid calibration count {calibration_count} for {len(train_episodes)} train episodes"
            )
        calibration_episodes = set(train_episodes[-calibration_count:])
        fit_episodes = set(train_episodes[:-calibration_count])
        fit = decisions_for_episodes(base, fit_episodes, meaningful)
        calibration = decisions_for_episodes(base, calibration_episodes, meaningful)
        weights = base.data["policy_weights"][fit].astype(np.float32)
        if not len(fit) or not len(calibration) or not len(holdout):
            raise Experiment7Error(
                f"{row['name']}: empty split fit={len(fit)} calibration={len(calibration)} holdout={len(holdout)}"
            )
        bundles.append((bundle, row))
        fit_sources.append((bundle, fit, weights))
        split_data[row["name"]] = {
            "fit": fit,
            "calibration": calibration,
            "holdout": holdout,
            "fitEpisodes": len(fit_episodes),
            "calibrationEpisodes": len(calibration_episodes),
            "holdoutEpisodes": len(set(int(value) for value in base.data["episode_ids"][holdout])),
        }
    if len(bundles) < 2:
        raise Experiment7Error("Experiment 7 multi-deck fine-tune requires at least two current sources")
    return bundles, fit_sources, split_data


def class_weights_for_fit(manifest: dict[str, Any], fit_sources, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    class_payload = read_json(Path(manifest["classMap"]["path"]))
    classes = len(class_payload["classes"])
    counts = np.zeros(classes, dtype=np.int64)
    for bundle, decisions, _ in fit_sources:
        labels = np.asarray(bundle.opponent_labels[decisions], dtype=np.int64)
        visible = np.asarray(bundle.opponent_visible_unique[decisions], dtype=np.int64)
        valid = (labels >= 0) & (visible > 0)
        counts += np.bincount(labels[valid], minlength=classes)
    # Auxiliary classification must never block the main policy. Empty classes
    # receive zero weight; populated classes preserve the reference inverse-sqrt weighting.
    weights = np.zeros(classes, dtype=np.float64)
    nonzero = counts > 0
    weights[nonzero] = 1.0 / np.sqrt(counts[nonzero].astype(np.float64))
    if np.any(nonzero):
        weights[nonzero] /= weights[nonzero].mean()
    return torch.from_numpy(weights.astype(np.float32)).to(device), counts


def model_config(vendor: dict[str, Any], manifest: dict[str, Any], args, opponent_classes: int):
    DeckIdentityModelConfig = vendor["DeckIdentityModelConfig"]
    return DeckIdentityModelConfig(
        card_vocab=int(manifest["engineCatalog"]["cardVocab"]),
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        ff_dim=args.ff_dim,
        history_length=8,
        opponent_classes=opponent_classes,
        dropout=args.dropout,
    )


def pretrain(args: argparse.Namespace) -> dict[str, Any]:
    manifest = source_manifest(args.sources.resolve())
    reference_root = Path(manifest["referenceRoot"])
    vendor = setup_vendor(reference_root)
    seed_everything(args.seed)
    device = device_from_arg(args.device)
    pretrain_bundle = load_bundle(vendor, "pretrain", manifest["pretrain"])
    decisions = np.flatnonzero(pretrain_bundle.sequence.base.nontrivial_mask())
    if args.max_decisions > 0:
        decisions = decisions[: args.max_decisions]
    catalog_path = Path(manifest["catalog"]["path"])
    module_by_episode = vendor["load_module_by_episode"](catalog_path)
    module_weights = vendor["MODULE_WEIGHTS"]
    weights = np.asarray(
        [
            float(pretrain_bundle.sequence.base.data["policy_weights"][decision])
            * float(module_weights.get(module_by_episode.get(int(pretrain_bundle.sequence.base.data["episode_ids"][decision]), ""), 1.0))
            for decision in decisions
        ],
        dtype=np.float32,
    )
    class_payload = read_json(Path(manifest["classMap"]["path"]))
    config = model_config(vendor, manifest, args, len(class_payload["classes"]))
    model = vendor["PTCGDeckIdentityTransformerPolicy"](config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    class_weights = torch.ones(config.opponent_classes, device=device)
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schemaVersion": 1,
        "stage": "shared_broad_pretrain",
        "createdAt": utc_now(),
        "sourcesSha256": sha256_file(args.sources),
        "seed": args.seed,
        "device": str(device),
        "modelConfig": config.to_dict(),
        "parameterCount": model.parameter_count,
        "decisions": int(len(decisions)),
        "epochs": [],
    }
    for epoch in range(1, args.epochs + 1):
        metrics = vendor["train_single_epoch"](
            model,
            pretrain_bundle,
            decisions,
            weights,
            optimizer,
            scaler,
            device,
            args.batch_size,
            rng,
            class_weights,
            0.0,
        )
        metrics["epoch"] = epoch
        report["epochs"].append(metrics)
        print(json.dumps({"stage": "pretrain", **metrics}), flush=True)
        vendor["save_checkpoint"](
            args.output_dir / "checkpoints" / f"pretrain_epoch_{epoch:02d}.pt",
            model,
            {"stage": "pretrain", "epoch": epoch, "metrics": metrics},
        )
    checkpoint = args.output_dir / "pretrain_model.pt"
    vendor["save_checkpoint"](
        checkpoint,
        model,
        {"stage": "pretrain", "epochs": args.epochs, "sourcesSha256": sha256_file(args.sources)},
    )
    report["checkpoint"] = {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)}
    write_json(args.output_dir / "pretrain_report.json", report)
    return report


def finetune(args: argparse.Namespace) -> dict[str, Any]:
    manifest = source_manifest(args.sources.resolve())
    reference_root = Path(manifest["referenceRoot"])
    vendor = setup_vendor(reference_root)
    seed_everything(args.seed)
    device = device_from_arg(args.device)
    bundles, fit_sources, split_data = current_splits(vendor, manifest)
    class_weights, class_counts = class_weights_for_fit(manifest, fit_sources, device)
    pretrain_checkpoint = load_checkpoint(args.pretrain_checkpoint.resolve(), device)
    config = vendor["DeckIdentityModelConfig"](**pretrain_checkpoint["config"])
    model = vendor["PTCGDeckIdentityTransformerPolicy"](config).to(device)
    model.load_state_dict(pretrain_checkpoint["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schemaVersion": 1,
        "stage": "balanced_multideck_finetune",
        "createdAt": utc_now(),
        "sourcesSha256": sha256_file(args.sources),
        "pretrainSha256": sha256_file(args.pretrain_checkpoint),
        "seed": args.seed,
        "device": str(device),
        "modelConfig": config.to_dict(),
        "parameterCount": model.parameter_count,
        "classFitCounts": class_counts.tolist(),
        "splits": {
            name: {
                "fitDecisions": int(len(value["fit"])),
                "calibrationDecisions": int(len(value["calibration"])),
                "holdoutDecisions": int(len(value["holdout"])),
                "fitEpisodes": value["fitEpisodes"],
                "calibrationEpisodes": value["calibrationEpisodes"],
                "holdoutEpisodes": value["holdoutEpisodes"],
            }
            for name, value in split_data.items()
        },
        "epochs": [],
    }
    best_score = -1.0
    best_epoch = 0
    best_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = vendor["train_balanced_multideck_epoch"](
            model,
            fit_sources,
            optimizer,
            scaler,
            device,
            args.batch_per_deck,
            rng,
            class_weights,
            args.opponent_loss_weight,
        )
        calibration = {
            bundle.name: vendor["evaluate_identity"](
                model,
                bundle,
                split_data[bundle.name]["calibration"],
                device,
                args.eval_batch_size,
            )
            for bundle, _ in bundles
        }
        scores = [float(value["exactSemantic"]) for value in calibration.values()]
        macro = float(np.mean(scores))
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "calibration": calibration,
            "calibrationMacroExactSemantic": macro,
            "calibrationWorstDeckExactSemantic": float(min(scores)),
            "calibrationDeckStd": float(np.std(scores)),
        }
        report["epochs"].append(row)
        print(json.dumps({"stage": "finetune", **row}), flush=True)
        if macro > best_score:
            best_score = macro
            best_epoch = epoch
            vendor["save_checkpoint"](
                best_path,
                model,
                {
                    "stage": "finetune",
                    "selectedFineTuneEpoch": epoch,
                    "calibration": calibration,
                    "macro": macro,
                    "seed": args.seed,
                },
            )
    checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    report["selectedEpoch"] = best_epoch
    report["selectedCalibrationMacroExactSemantic"] = best_score
    report["selectedCalibration"] = {
        bundle.name: vendor["evaluate_identity"](
            model,
            bundle,
            split_data[bundle.name]["calibration"],
            device,
            args.eval_batch_size,
        )
        for bundle, _ in bundles
    }
    report["checkpoint"] = {"path": str(best_path.resolve()), "sha256": sha256_file(best_path)}
    write_json(args.output_dir / "finetune_report.json", report)
    return report


def holdout(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"holdout receipt already exists and is sealed: {args.output}")
    manifest = source_manifest(args.sources.resolve())
    vendor = setup_vendor(Path(manifest["referenceRoot"]))
    device = device_from_arg(args.device)
    bundles, _, split_data = current_splits(vendor, manifest)
    checkpoint = load_checkpoint(args.checkpoint.resolve(), device)
    config = vendor["DeckIdentityModelConfig"](**checkpoint["config"])
    model = vendor["PTCGDeckIdentityTransformerPolicy"](config).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    result = {
        bundle.name: vendor["evaluate_identity"](
            model,
            bundle,
            split_data[bundle.name]["holdout"],
            device,
            args.batch_size,
        )
        for bundle, _ in bundles
    }
    scores = [float(value["exactSemantic"]) for value in result.values()]
    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "holdoutOpened": True,
        "oneShotReceipt": True,
        "sources": {"path": str(args.sources.resolve()), "sha256": sha256_file(args.sources)},
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256_file(args.checkpoint)},
        "perDeck": result,
        "macroExactSemantic": float(np.mean(scores)),
        "worstDeckExactSemantic": float(min(scores)),
        "deckStd": float(np.std(scores)),
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    manifest = source_manifest(args.sources.resolve())
    vendor = setup_vendor(Path(manifest["referenceRoot"]))
    seed_everything(args.seed)
    device = device_from_arg(args.device)
    bundles, fit_sources, split_data = current_splits(vendor, manifest)
    class_weights, _ = class_weights_for_fit(manifest, fit_sources, device)
    class_payload = read_json(Path(manifest["classMap"]["path"]))
    config = model_config(vendor, manifest, args, len(class_payload["classes"]))
    model = vendor["PTCGDeckIdentityTransformerPolicy"](config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(args.seed)
    tiny_sources = []
    for bundle, decisions, weights in fit_sources:
        take = min(args.tiny_decisions_per_deck, len(decisions))
        tiny_sources.append((bundle, decisions[:take], weights[:take]))
    initial = {
        bundle.name: vendor["evaluate_identity"](
            model, bundle, decisions, device, args.batch_per_deck
        )
        for bundle, decisions, _ in tiny_sources
    }
    history = []
    for epoch in range(1, args.epochs + 1):
        metrics = vendor["train_balanced_multideck_epoch"](
            model,
            tiny_sources,
            optimizer,
            scaler,
            device,
            args.batch_per_deck,
            rng,
            class_weights,
            args.opponent_loss_weight,
        )
        history.append({"epoch": epoch, "train": metrics})
    final = {
        bundle.name: vendor["evaluate_identity"](
            model, bundle, decisions, device, args.batch_per_deck
        )
        for bundle, decisions, _ in tiny_sources
    }
    initial_macro = float(np.mean([value["exactSemantic"] for value in initial.values()]))
    final_macro = float(np.mean([value["exactSemantic"] for value in final.values()]))
    illegal = sum(int(value["illegalPredictionCount"]) for value in final.values())
    checkpoint = args.output_dir / "smoke_model.pt"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vendor["save_checkpoint"](checkpoint, model, {"stage": "smoke", "final": final})
    reloaded = vendor["PTCGDeckIdentityTransformerPolicy"](config).to(device)
    reloaded.load_state_dict(load_checkpoint(checkpoint, device)["state_dict"], strict=True)
    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "tinyDecisionsPerDeck": args.tiny_decisions_per_deck,
        "epochs": args.epochs,
        "initial": initial,
        "final": final,
        "initialMacroExactSemantic": initial_macro,
        "finalMacroExactSemantic": final_macro,
        "improvement": final_macro - initial_macro,
        "illegalPredictionCount": illegal,
        "checkpointReloadedStrictly": True,
        "passes": illegal == 0 and final_macro >= args.min_exact_semantic and final_macro > initial_macro,
    }
    write_json(args.output_dir / "smoke_report.json", payload)
    if not payload["passes"]:
        raise Experiment7Error(json.dumps(payload, ensure_ascii=False))
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 7 shared-pretrain and multi-deck fine-tune driver")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pre = subparsers.add_parser("pretrain")
    pre.add_argument("--sources", type=Path, required=True)
    pre.add_argument("--output-dir", type=Path, required=True)
    pre.add_argument("--seed", type=int, default=20260808)
    pre.add_argument("--device", default="auto")
    pre.add_argument("--epochs", type=int, default=12)
    pre.add_argument("--batch-size", type=int, default=128)
    pre.add_argument("--learning-rate", type=float, default=3e-4)
    pre.add_argument("--weight-decay", type=float, default=1e-4)
    pre.add_argument("--max-decisions", type=int, default=0)
    add_model_args(pre)

    fine = subparsers.add_parser("finetune")
    fine.add_argument("--sources", type=Path, required=True)
    fine.add_argument("--pretrain-checkpoint", type=Path, required=True)
    fine.add_argument("--output-dir", type=Path, required=True)
    fine.add_argument("--seed", type=int, required=True)
    fine.add_argument("--device", default="auto")
    fine.add_argument("--epochs", type=int, default=6)
    fine.add_argument("--batch-per-deck", type=int, default=48)
    fine.add_argument("--eval-batch-size", type=int, default=128)
    fine.add_argument("--learning-rate", type=float, default=1e-4)
    fine.add_argument("--weight-decay", type=float, default=1e-4)
    fine.add_argument("--opponent-loss-weight", type=float, default=0.05)

    hold = subparsers.add_parser("holdout")
    hold.add_argument("--sources", type=Path, required=True)
    hold.add_argument("--checkpoint", type=Path, required=True)
    hold.add_argument("--output", type=Path, required=True)
    hold.add_argument("--device", default="auto")
    hold.add_argument("--batch-size", type=int, default=128)

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--sources", type=Path, required=True)
    smoke_parser.add_argument("--output-dir", type=Path, required=True)
    smoke_parser.add_argument("--seed", type=int, default=20260808)
    smoke_parser.add_argument("--device", default="auto")
    smoke_parser.add_argument("--epochs", type=int, default=30)
    smoke_parser.add_argument("--tiny-decisions-per-deck", type=int, default=64)
    smoke_parser.add_argument("--batch-per-deck", type=int, default=32)
    smoke_parser.add_argument("--learning-rate", type=float, default=1e-3)
    smoke_parser.add_argument("--opponent-loss-weight", type=float, default=0.05)
    smoke_parser.add_argument("--min-exact-semantic", type=float, default=0.70)
    add_model_args(smoke_parser)

    args = parser.parse_args()
    if args.command == "pretrain":
        pretrain(args)
    elif args.command == "finetune":
        finetune(args)
    elif args.command == "holdout":
        holdout(args)
    elif args.command == "smoke":
        smoke(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
