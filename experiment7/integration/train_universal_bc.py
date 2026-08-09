from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import Experiment7Error, read_json, sha256_file, utc_now, write_json


def setup_vendor(reference_root: Path) -> dict[str, Any]:
    for path in (reference_root / "training", reference_root / "data_pipeline"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from train import batches
    from train_multideck_identity import IdentityBundle, make_identity_batch
    from universal_deck_model import (
        UniversalDeckModelConfig,
        UniversalDeckTransformerPolicy,
        universal_bc_loss,
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


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def migrate_compatible_weights(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    payload = load_checkpoint(checkpoint, device)
    source = payload.get("state_dict", payload)
    target = model.state_dict()
    compatible = {
        name: value
        for name, value in source.items()
        if name in target and tuple(value.shape) == tuple(target[name].shape)
    }
    result = model.load_state_dict(compatible, strict=False)
    return {
        "path": str(checkpoint.resolve()),
        "sha256": sha256_file(checkpoint),
        "loadedTensors": len(compatible),
        "loadedParameters": int(sum(value.numel() for value in compatible.values())),
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
    }


def forward_model(model, batch: dict[str, torch.Tensor]):
    return model(
        batch["state"],
        batch["history_state"],
        batch["history_action"],
        batch["history_mask"],
        batch["own_deck_cards"],
        batch["entity_cat"],
        batch["entity_num"],
        batch["entity_mask"],
        batch["options"],
        batch["option_mask"],
    )


def save_checkpoint(path: Path, model: torch.nn.Module, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 2,
            "architecture": "experiment7_universal_deck8_autoregressive_stop",
            "config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def train_epoch(
    vendor: dict[str, Any],
    model,
    bundle,
    decisions: np.ndarray,
    policy_weight_by_decision: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    batch_size: int,
    rng: np.random.Generator,
    value_loss_weight: float,
) -> dict[str, float]:
    model.train()
    totals: Counter[str] = Counter()
    examples = 0
    started = time.perf_counter()
    amp = device.type == "cuda"
    for decision_batch in vendor["batches"](decisions, batch_size, rng):
        batch = vendor["make_identity_batch"](bundle, decision_batch, device)
        policy_weights = torch.from_numpy(
            policy_weight_by_decision[decision_batch].astype(np.float32, copy=True)
        ).to(device)
        # Value is deliberately trained from both seats, independently of BC policy weight.
        value_weights = torch.ones(len(decision_batch), dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            encoding = forward_model(model, batch)
            loss, parts = vendor["universal_bc_loss"](
                model,
                encoding,
                batch,
                policy_weights,
                value_weights,
                value_loss_weight=value_loss_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        count = len(decision_batch)
        examples += count
        for key, value in parts.items():
            totals[key] += float(value.detach().cpu()) * count
    elapsed = time.perf_counter() - started
    return {
        **{key: float(value / max(examples, 1)) for key, value in totals.items()},
        "decisions": examples,
        "seconds": elapsed,
        "decisionsPerSecond": examples / max(elapsed, 1e-9),
    }


@torch.inference_mode()
def evaluate(
    vendor: dict[str, Any],
    model,
    bundle,
    decisions: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    base = bundle.sequence.base
    exact_index: list[int] = []
    exact_semantic: list[int] = []
    count_correct: list[int] = []
    first_step_confidence: list[float] = []
    value_predictions: list[float] = []
    value_targets: list[float] = []
    policy_examples = 0
    illegal = 0
    for decision_batch in vendor["batches"](decisions, batch_size, None):
        batch = vendor["make_identity_batch"](bundle, decision_batch, device)
        encoding = forward_model(model, batch)
        predictions = model.greedy_actions(
            encoding, batch["min_count"], batch["max_count"]
        )
        empty_selected = torch.zeros_like(batch["option_mask"], dtype=torch.bool)
        first_logits = model.decoder_logits(
            encoding, empty_selected, batch["min_count"], batch["max_count"]
        )
        first_probabilities = torch.softmax(first_logits.float(), dim=1)
        first_step_confidence.extend(
            first_probabilities.max(dim=1).values.cpu().numpy().tolist()
        )
        value_predictions.extend(
            torch.sigmoid(encoding.value_logits.float()).cpu().numpy().tolist()
        )
        value_targets.extend(batch["winner"].float().cpu().numpy().tolist())

        for row, decision_value in enumerate(decision_batch):
            decision = int(decision_value)
            if float(base.data["policy_weights"][decision]) <= 0:
                continue
            policy_examples += 1
            begin = int(base.data["option_offsets"][decision])
            end = int(base.data["option_offsets"][decision + 1])
            option_count = end - begin
            predicted = predictions[row]
            expert = [
                int(value)
                for value in np.flatnonzero(base.data["option_labels"][begin:end])
            ]
            minimum = int(base.data["min_counts"][decision])
            maximum = int(base.data["max_counts"][decision])
            if (
                len(predicted) < minimum
                or len(predicted) > maximum
                or len(set(predicted)) != len(predicted)
                or any(index < 0 or index >= option_count for index in predicted)
            ):
                illegal += 1
            exact_index.append(int(set(predicted) == set(expert)))
            hashes = np.asarray(base.semantic_hash[decision, :option_count], dtype=np.uint32)
            predicted_semantic = Counter(int(hashes[index]) for index in predicted)
            expert_semantic = Counter(int(hashes[index]) for index in expert)
            exact_semantic.append(int(predicted_semantic == expert_semantic))
            count_correct.append(int(len(predicted) == len(expert)))

    values = np.asarray(value_predictions, dtype=np.float64)
    targets = np.asarray(value_targets, dtype=np.float64)
    confidence = np.asarray(first_step_confidence, dtype=np.float64)
    return {
        "decisions": int(len(decisions)),
        "policyDecisions": policy_examples,
        "exactIndex": float(np.mean(exact_index)) if exact_index else None,
        "exactSemantic": float(np.mean(exact_semantic)) if exact_semantic else None,
        "countAccuracy": float(np.mean(count_correct)) if count_correct else None,
        "illegalPredictionCount": illegal,
        "valueBrier": float(np.mean((values - targets) ** 2)) if len(values) else None,
        "uncertainty": {
            "meanFirstStepConfidence": float(confidence.mean()) if len(confidence) else None,
            "confidence60Coverage": float(np.mean(confidence >= 0.60)) if len(confidence) else None,
        },
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    sources = read_json(args.sources.resolve())
    if sources.get("kind") != "experiment7_universal_bc":
        raise Experiment7Error("not a Universal BC source manifest")
    reference_root = Path(sources["referenceRoot"])
    vendor = setup_vendor(reference_root)
    seed_everything(args.seed)
    device = device_from_arg(args.device)
    row = sources["dataset"]
    bundle = vendor["IdentityBundle"].load(
        "universal",
        Path(row["features"]),
        Path(row["tokenCache"]),
        Path(row["sequenceCache"]),
        Path(row["identityCache"]),
    )
    base = bundle.sequence.base
    meaningful = base.nontrivial_mask()
    train_decisions = np.flatnonzero((base.data["validation"] == 0) & meaningful)
    validation_decisions = np.flatnonzero((base.data["validation"] == 1) & meaningful)
    if args.max_train_decisions > 0:
        train_decisions = train_decisions[: args.max_train_decisions]
    if args.max_validation_decisions > 0:
        validation_decisions = validation_decisions[: args.max_validation_decisions]
    if not len(train_decisions) or not len(validation_decisions):
        raise Experiment7Error(
            f"empty Universal BC split train={len(train_decisions)} validation={len(validation_decisions)}"
        )

    config = vendor["UniversalDeckModelConfig"](
        card_vocab=int(sources["engineCatalog"]["cardVocab"]),
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        ff_dim=args.ff_dim,
        history_length=8,
        deck_latents=8,
        opponent_classes=1,
        dropout=args.dropout,
    )
    model = vendor["UniversalDeckTransformerPolicy"](config).to(device)
    initialization = None
    if args.initialize_from:
        initialization = migrate_compatible_weights(
            model, args.initialize_from.resolve(), device
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(args.seed)
    policy_weight_by_decision = np.asarray(
        base.data["policy_weights"], dtype=np.float32
    ).copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schemaVersion": 2,
        "stage": "universal_bc",
        "createdAt": utc_now(),
        "sources": {"path": str(args.sources.resolve()), "sha256": sha256_file(args.sources)},
        "seed": args.seed,
        "device": str(device),
        "modelConfig": config.to_dict(),
        "parameterCount": model.parameter_count,
        "initialization": initialization,
        "splits": {
            "trainDecisions": int(len(train_decisions)),
            "trainPolicyDecisions": int(np.sum(policy_weight_by_decision[train_decisions] > 0)),
            "validationDecisions": int(len(validation_decisions)),
            "validationPolicyDecisions": int(np.sum(policy_weight_by_decision[validation_decisions] > 0)),
        },
        "epochs": [],
    }
    best_score = -1.0
    best_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(
            vendor,
            model,
            bundle,
            train_decisions,
            policy_weight_by_decision,
            optimizer,
            scaler,
            device,
            args.batch_size,
            rng,
            args.value_loss_weight,
        )
        validation = evaluate(
            vendor,
            model,
            bundle,
            validation_decisions,
            device,
            args.batch_size,
        )
        epoch_row = {"epoch": epoch, "training": metrics, "validation": validation}
        report["epochs"].append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False), flush=True)
        epoch_path = args.output_dir / "checkpoints" / f"epoch_{epoch:02d}.pt"
        save_checkpoint(epoch_path, model, epoch_row)
        score = float(validation.get("exactSemantic") or 0.0)
        if score > best_score:
            best_score = score
            save_checkpoint(
                best_path,
                model,
                {"epoch": epoch, "validation": validation, "sourcesSha256": sha256_file(args.sources)},
            )
            report["best"] = {
                "epoch": epoch,
                "score": score,
                "path": str(best_path.resolve()),
                "sha256": sha256_file(best_path),
            }
        write_json(args.output_dir / "training_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Experiment 7 Universal Deck-8 BC")
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.05)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ff-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-train-decisions", type=int, default=0)
    parser.add_argument("--max-validation-decisions", type=int, default=0)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
