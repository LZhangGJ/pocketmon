from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import Experiment7Error, read_json, sha256_file, utc_now, write_json
from feature_tensor_store import install_memmap_bundle_loader


def setup_vendor(reference_root: Path) -> dict[str, Any]:
    for path in (reference_root / "training", reference_root / "data_pipeline"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from train import Bundle, batches
    install_memmap_bundle_loader(Bundle)
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


def _pin_tensor_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Pin one bounded CPU batch so CUDA copies can overlap model compute."""
    return {
        name: tensor.pin_memory() if tensor.device.type == "cpu" else tensor
        for name, tensor in batch.items()
    }


def _copy_tensor_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.to(device, non_blocking=device.type == "cuda")
        for name, tensor in batch.items()
    }


def prefetched_identity_batches(
    vendor: dict[str, Any],
    bundle,
    decisions: np.ndarray,
    policy_weight_by_decision: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
    prefetch_batches: int,
    prefetch_workers: int,
):
    """Build a small ordered queue in RAM while the GPU trains the prior batch.

    The queue is deliberately bounded.  It accelerates mmap/RAM-cache gathers and
    Python collation without materializing the full replay corpus in either RAM
    or VRAM, and preserves the exact batch order produced by ``batches``.
    """
    decision_iterator = iter(vendor["batches"](decisions, batch_size, rng))
    cpu = torch.device("cpu")

    def build(decision_batch: np.ndarray):
        batch = vendor["make_identity_batch"](bundle, decision_batch, cpu)
        weights = torch.from_numpy(
            policy_weight_by_decision[decision_batch].astype(np.float32, copy=True)
        )
        if device.type == "cuda":
            batch = _pin_tensor_batch(batch)
            weights = weights.pin_memory()
        return decision_batch, batch, weights

    with ThreadPoolExecutor(max_workers=prefetch_workers) as executor:
        pending: deque[Future] = deque()
        for _ in range(prefetch_batches):
            try:
                pending.append(executor.submit(build, next(decision_iterator)))
            except StopIteration:
                break
        while pending:
            decision_batch, cpu_batch, cpu_weights = pending.popleft().result()
            try:
                pending.append(executor.submit(build, next(decision_iterator)))
            except StopIteration:
                pass
            yield (
                decision_batch,
                _copy_tensor_batch(cpu_batch, device),
                cpu_weights.to(device, non_blocking=device.type == "cuda"),
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
    prefetch_batches: int = 0,
    prefetch_workers: int = 1,
) -> dict[str, float]:
    model.train()
    totals: Counter[str] = Counter()
    examples = 0
    started = time.perf_counter()
    amp = device.type == "cuda"
    if prefetch_batches > 0:
        batch_iterator = prefetched_identity_batches(
            vendor,
            bundle,
            decisions,
            policy_weight_by_decision,
            batch_size,
            rng,
            device,
            prefetch_batches,
            prefetch_workers,
        )
    else:
        batch_iterator = (
            (
                decision_batch,
                vendor["make_identity_batch"](bundle, decision_batch, device),
                torch.from_numpy(
                    policy_weight_by_decision[decision_batch].astype(
                        np.float32, copy=True
                    )
                ).to(device),
            )
            for decision_batch in vendor["batches"](decisions, batch_size, rng)
        )
    for decision_batch, batch, policy_weights in batch_iterator:
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


def prepare_shards(
    sources: dict[str, Any],
    vendor: dict[str, Any],
    validation_fraction: float,
    max_train_decisions: int,
    max_validation_decisions: int,
) -> list[dict[str, Any]]:
    rows = sources.get("datasets")
    if rows is None:
        rows = [sources["dataset"]]
    if not isinstance(rows, list) or not rows:
        raise Experiment7Error("Universal BC sources must contain a non-empty dataset(s) list")

    names: set[str] = set()
    shards: list[dict[str, Any]] = []
    train_limits = _distributed_limits(max_train_decisions, len(rows))
    validation_limits = _distributed_limits(max_validation_decisions, len(rows))
    for index, row in enumerate(rows):
        name = str(row.get("name") or f"universal-{index:02d}")
        if name in names:
            raise Experiment7Error(f"duplicate Universal BC dataset name: {name}")
        names.add(name)
        bundle = vendor["IdentityBundle"].load(
            name,
            Path(row["features"]),
            Path(row["tokenCache"]),
            Path(row["sequenceCache"]),
            Path(row["identityCache"]),
        )
        base = bundle.sequence.base
        meaningful = base.nontrivial_mask()
        validation_mask = base.data["validation"] == 1
        if not bool(validation_mask.any()):
            episode_ids = np.asarray(base.data["episode_ids"], dtype=np.int64)
            ordered_episodes = list(dict.fromkeys(int(value) for value in episode_ids))
            validation_count = max(
                1, int(math.ceil(len(ordered_episodes) * validation_fraction))
            )
            validation_episodes = set(ordered_episodes[-validation_count:])
            validation_mask = np.isin(episode_ids, list(validation_episodes))
        train_decisions = np.flatnonzero(~validation_mask & meaningful)
        validation_decisions = np.flatnonzero(validation_mask & meaningful)
        if train_limits[index] is not None:
            train_decisions = train_decisions[: train_limits[index]]
        if validation_limits[index] is not None:
            validation_decisions = validation_decisions[: validation_limits[index]]
        if not len(train_decisions) or not len(validation_decisions):
            raise Experiment7Error(
                f"{name}: empty Universal BC split "
                f"train={len(train_decisions)} validation={len(validation_decisions)}"
            )
        policy_weights = np.asarray(
            base.data["policy_weights"], dtype=np.float32
        ).copy()
        shards.append(
            {
                "name": name,
                "bundle": bundle,
                "train": train_decisions,
                "validation": validation_decisions,
                "policyWeights": policy_weights,
            }
        )
    return shards


def _distributed_limits(total: int, count: int) -> list[int | None]:
    if total <= 0:
        return [None] * count
    if total < count:
        raise Experiment7Error(
            f"decision limit {total} is smaller than Universal BC shard count {count}"
        )
    quotient, remainder = divmod(total, count)
    return [quotient + int(index < remainder) for index in range(count)]


def combine_training_metrics(rows: list[tuple[str, dict[str, float]]]) -> dict[str, Any]:
    decisions = sum(int(row["decisions"]) for _, row in rows)
    seconds = sum(float(row["seconds"]) for _, row in rows)
    ignored = {"decisions", "seconds", "decisionsPerSecond"}
    keys = sorted({key for _, row in rows for key in row if key not in ignored})
    return {
        **{
            key: float(
                sum(float(row[key]) * int(row["decisions"]) for _, row in rows)
                / max(decisions, 1)
            )
            for key in keys
        },
        "decisions": decisions,
        "seconds": seconds,
        "decisionsPerSecond": decisions / max(seconds, 1e-9),
        "shards": {name: row for name, row in rows},
    }


def combine_validation_metrics(rows: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    decisions = sum(int(row["decisions"]) for _, row in rows)
    policy_decisions = sum(int(row["policyDecisions"]) for _, row in rows)

    def weighted(key: str, weight_key: str) -> float | None:
        eligible = [
            (float(row[key]), int(row[weight_key]))
            for _, row in rows
            if row.get(key) is not None and int(row[weight_key]) > 0
        ]
        denominator = sum(weight for _, weight in eligible)
        return (
            sum(value * weight for value, weight in eligible) / denominator
            if denominator
            else None
        )

    def uncertainty_weighted(key: str) -> float | None:
        eligible = [
            (float(row["uncertainty"][key]), int(row["decisions"]))
            for _, row in rows
            if row["uncertainty"].get(key) is not None and int(row["decisions"]) > 0
        ]
        denominator = sum(weight for _, weight in eligible)
        return (
            sum(value * weight for value, weight in eligible) / denominator
            if denominator
            else None
        )

    return {
        "decisions": decisions,
        "policyDecisions": policy_decisions,
        "exactIndex": weighted("exactIndex", "policyDecisions"),
        "exactSemantic": weighted("exactSemantic", "policyDecisions"),
        "countAccuracy": weighted("countAccuracy", "policyDecisions"),
        "illegalPredictionCount": sum(
            int(row["illegalPredictionCount"]) for _, row in rows
        ),
        "valueBrier": weighted("valueBrier", "decisions"),
        "uncertainty": {
            "meanFirstStepConfidence": uncertainty_weighted(
                "meanFirstStepConfidence"
            ),
            "confidence60Coverage": uncertainty_weighted("confidence60Coverage"),
        },
        "shards": {name: row for name, row in rows},
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    sources = read_json(args.sources.resolve())
    universal_sources = sources.get("kind") == "experiment7_universal_bc"
    if not universal_sources and not args.allow_multideck_pretrain:
        raise Experiment7Error(
            "not a Universal BC source manifest; use --allow-multideck-pretrain only for bootstrap"
        )
    reference_root = Path(sources["referenceRoot"])
    vendor = setup_vendor(reference_root)
    seed_everything(args.seed)
    device = device_from_arg(args.device)
    if universal_sources:
        shards = prepare_shards(
            sources,
            vendor,
            args.validation_fraction,
            args.max_train_decisions,
            args.max_validation_decisions,
        )
    else:
        bootstrap_sources = {**sources, "dataset": sources["pretrain"]}
        shards = prepare_shards(
            bootstrap_sources,
            vendor,
            args.validation_fraction,
            args.max_train_decisions,
            args.max_validation_decisions,
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
    learning_rate_schedule: list[tuple[int, float]] = []
    for value in args.learning_rate_schedule:
        epoch_text, rate_text = value.split("=", 1)
        learning_rate_schedule.append((int(epoch_text), float(rate_text)))
    learning_rate_schedule.sort()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schemaVersion": 3,
        "stage": "universal_bc",
        "bootstrapFromMultideckPretrain": not universal_sources,
        "createdAt": utc_now(),
        "sources": {"path": str(args.sources.resolve()), "sha256": sha256_file(args.sources)},
        "seed": args.seed,
        "device": str(device),
        "modelConfig": config.to_dict(),
        "parameterCount": model.parameter_count,
        "initialization": initialization,
        "splits": {
            "trainDecisions": int(sum(len(shard["train"]) for shard in shards)),
            "trainPolicyDecisions": int(
                sum(
                    np.sum(shard["policyWeights"][shard["train"]] > 0)
                    for shard in shards
                )
            ),
            "validationDecisions": int(
                sum(len(shard["validation"]) for shard in shards)
            ),
            "validationPolicyDecisions": int(
                sum(
                    np.sum(shard["policyWeights"][shard["validation"]] > 0)
                    for shard in shards
                )
            ),
            "shards": {
                shard["name"]: {
                    "trainDecisions": int(len(shard["train"])),
                    "trainPolicyDecisions": int(
                        np.sum(shard["policyWeights"][shard["train"]] > 0)
                    ),
                    "validationDecisions": int(len(shard["validation"])),
                    "validationPolicyDecisions": int(
                        np.sum(shard["policyWeights"][shard["validation"]] > 0)
                    ),
                }
                for shard in shards
            },
        },
        "epochs": [],
    }
    early_stop_best = -1.0
    early_stop_previous_brier: float | None = None
    early_stop_stagnant = 0
    early_stop_brier_regressions = 0
    if args.early_stop_baseline_report:
        baseline_report = read_json(args.early_stop_baseline_report)
        baseline_validation = baseline_report["epochs"][-1]["validation"]
        early_stop_best = float(baseline_validation.get("exactSemantic") or 0.0)
        early_stop_previous_brier = float(baseline_validation.get("valueBrier") or 0.0)
        report["earlyStopping"] = {
            "baselineReport": str(args.early_stop_baseline_report.resolve()),
            "baselineExactSemantic": early_stop_best,
            "baselineValueBrier": early_stop_previous_brier,
            "patience": args.early_stop_patience,
            "minSemanticDelta": args.early_stop_min_semantic_delta,
            "maxBrierIncrease": args.early_stop_max_brier_increase,
        }
    print(
        json.dumps(
            {
                "stage": "start",
                "device": str(device),
                "parameterCount": model.parameter_count,
                "splits": report["splits"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    best_score = -1.0
    best_path = args.output_dir / "best_model.pt"
    for epoch in range(args.epoch_start, args.epoch_start + args.epochs):
        learning_rate = args.learning_rate
        for milestone_epoch, milestone_rate in learning_rate_schedule:
            if epoch >= milestone_epoch:
                learning_rate = milestone_rate
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        training_rows: list[tuple[str, dict[str, float]]] = []
        for shard_index in rng.permutation(len(shards)):
            shard = shards[int(shard_index)]
            shard_metrics = train_epoch(
                vendor,
                model,
                shard["bundle"],
                shard["train"],
                shard["policyWeights"],
                optimizer,
                scaler,
                device,
                args.batch_size,
                rng,
                args.value_loss_weight,
                args.prefetch_batches,
                args.prefetch_workers,
            )
            training_rows.append((shard["name"], shard_metrics))
            print(
                json.dumps(
                    {
                        "stage": "train_shard",
                        "epoch": epoch,
                        "shard": shard["name"],
                        **shard_metrics,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        metrics = combine_training_metrics(training_rows)
        validation_rows: list[tuple[str, dict[str, Any]]] = []
        for shard in shards:
            shard_validation = evaluate(
                vendor,
                model,
                shard["bundle"],
                shard["validation"],
                device,
                args.batch_size,
            )
            validation_rows.append((shard["name"], shard_validation))
            print(
                json.dumps(
                    {
                        "stage": "validation_shard",
                        "epoch": epoch,
                        "shard": shard["name"],
                        **shard_validation,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        validation = combine_validation_metrics(validation_rows)
        epoch_row = {
            "epoch": epoch,
            "learningRate": learning_rate,
            "training": metrics,
            "validation": validation,
        }
        report["epochs"].append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False), flush=True)
        epoch_path = args.output_dir / "checkpoints" / f"epoch_{epoch:02d}.pt"
        save_checkpoint(epoch_path, model, epoch_row)
        score = float(validation.get("exactSemantic") or 0.0)
        value_brier = float(validation.get("valueBrier") or 0.0)
        if args.early_stop_patience > 0:
            if score >= early_stop_best + args.early_stop_min_semantic_delta:
                early_stop_best = score
                early_stop_stagnant = 0
            else:
                early_stop_stagnant += 1
            if (
                early_stop_previous_brier is not None
                and value_brier
                > early_stop_previous_brier + args.early_stop_max_brier_increase
            ):
                early_stop_brier_regressions += 1
            else:
                early_stop_brier_regressions = 0
            early_stop_previous_brier = value_brier
            epoch_row["earlyStopping"] = {
                "bestExactSemantic": early_stop_best,
                "stagnantEpochs": early_stop_stagnant,
                "brierRegressionEpochs": early_stop_brier_regressions,
            }
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
        if args.early_stop_patience > 0 and (
            early_stop_stagnant >= args.early_stop_patience
            or early_stop_brier_regressions >= args.early_stop_patience
        ):
            report["earlyStopping"].update(
                {
                    "stopped": True,
                    "stoppedAtEpoch": epoch,
                    "reason": (
                        "semantic_plateau"
                        if early_stop_stagnant >= args.early_stop_patience
                        else "value_brier_regression"
                    ),
                }
            )
            write_json(args.output_dir / "training_report.json", report)
            break
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Experiment 7 Universal Deck-8 BC")
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument(
        "--allow-multideck-pretrain",
        action="store_true",
        help="Bootstrap Deck-8/STOP weights from the prior broad winner-only cache",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.05,
        help="Chronological split used only when the source cache has no validation rows",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--epoch-start",
        type=int,
        default=1,
        help="Absolute epoch number assigned to the first epoch in this persistent run",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--prefetch-batches",
        type=int,
        default=0,
        help="Bounded number of CPU/pinned-RAM batches prepared ahead of CUDA compute",
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=1,
        help="CPU collation workers used by --prefetch-batches",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--learning-rate-schedule",
        action="append",
        default=[],
        metavar="EPOCH=RATE",
        help="Set the learning rate from an absolute epoch onward; may be repeated",
    )
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-semantic-delta", type=float, default=0.002)
    parser.add_argument("--early-stop-max-brier-increase", type=float, default=0.005)
    parser.add_argument("--early-stop-baseline-report", type=Path)
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
    if args.prefetch_batches < 0:
        parser.error("--prefetch-batches must be non-negative")
    if args.prefetch_workers < 1:
        parser.error("--prefetch-workers must be at least one")
    if args.epoch_start < 1:
        parser.error("--epoch-start must be at least one")
    if args.early_stop_patience < 0:
        parser.error("--early-stop-patience must be non-negative")
    for value in args.learning_rate_schedule:
        try:
            epoch_text, rate_text = value.split("=", 1)
            if int(epoch_text) < 1 or float(rate_text) <= 0:
                raise ValueError
        except ValueError:
            parser.error(f"invalid --learning-rate-schedule value: {value!r}")
    train(args)


if __name__ == "__main__":
    main()
