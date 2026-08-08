from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from deck_identity_model import DeckIdentityModelConfig, PTCGDeckIdentityTransformerPolicy
from sequence_model import PTCGSequenceTransformerPolicy, SequenceModelConfig
from train import (
    MODULE_WEIGHTS,
    batches,
    decisions_for_episodes,
    episode_order,
    load_module_by_episode,
    loss_terms,
    masked_count_logits,
)
from train_sequence import SequenceBundle, evaluate as evaluate_sequence, make_sequence_batch


@dataclass
class IdentityBundle:
    name: str
    sequence: SequenceBundle
    own_deck_cards: np.ndarray
    opponent_labels: np.ndarray
    opponent_visible_unique: np.ndarray

    @classmethod
    def load(
        cls,
        name: str,
        features: Path,
        token_cache: Path,
        sequence_cache: Path,
        identity_cache: Path,
    ) -> "IdentityBundle":
        sequence = SequenceBundle.load(features, token_cache, sequence_cache)
        own_cards = np.load(identity_cache / "own_deck_cards.npy", mmap_mode="r")
        labels = np.load(identity_cache / "opponent_deck_labels.npy", mmap_mode="r")
        visible = np.load(
            identity_cache / "opponent_visible_unique_cards.npy", mmap_mode="r"
        )
        decisions = len(sequence.base.data["episode_ids"])
        if own_cards.shape != (decisions, 60):
            raise RuntimeError(f"{name}: own deck cache shape mismatch")
        if labels.shape != (decisions,) or visible.shape != (decisions,):
            raise RuntimeError(f"{name}: opponent label cache shape mismatch")
        return cls(name, sequence, own_cards, labels, visible)


def make_identity_batch(
    bundle: IdentityBundle, decisions: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    result = make_sequence_batch(bundle.sequence, decisions, device)
    result["own_deck_cards"] = torch.from_numpy(
        np.asarray(bundle.own_deck_cards[decisions], dtype=np.int64).copy()
    ).to(device)
    result["opponent_label"] = torch.from_numpy(
        np.asarray(bundle.opponent_labels[decisions], dtype=np.int64).copy()
    ).to(device)
    result["opponent_visible_unique"] = torch.from_numpy(
        np.asarray(bundle.opponent_visible_unique[decisions], dtype=np.int64).copy()
    ).to(device)
    return result


def forward_identity(
    model: PTCGDeckIdentityTransformerPolicy, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def weighted_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    batch: dict[str, torch.Tensor],
    decision_weights: torch.Tensor,
    class_weights: torch.Tensor,
    opponent_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    option_logits, count_logits, value_logits, opponent_logits = outputs
    base_loss, parts = loss_terms(
        option_logits, count_logits, value_logits, batch, decision_weights
    )
    valid = (batch["opponent_label"] >= 0) & (batch["opponent_visible_unique"] > 0)
    if opponent_loss_weight > 0 and bool(valid.any()):
        opponent_loss = F.cross_entropy(
            opponent_logits[valid], batch["opponent_label"][valid], weight=class_weights
        )
    else:
        opponent_loss = opponent_logits.sum() * 0.0
    total = base_loss + opponent_loss_weight * opponent_loss
    return total, {
        **parts,
        "opponent": float(opponent_loss.detach().cpu()),
        "opponentExamples": int(valid.sum().detach().cpu()),
        "totalWithOpponent": float(total.detach().cpu()),
    }


def train_single_epoch(
    model: PTCGDeckIdentityTransformerPolicy,
    bundle: IdentityBundle,
    decisions: np.ndarray,
    weights: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    batch_size: int,
    rng: np.random.Generator,
    class_weights: torch.Tensor,
    opponent_loss_weight: float,
) -> dict[str, float]:
    model.train()
    weight_by_decision = np.zeros(
        len(bundle.sequence.base.data["episode_ids"]), dtype=np.float32
    )
    weight_by_decision[decisions] = weights
    totals: Counter[str] = Counter()
    examples = 0
    started = time.perf_counter()
    amp = device.type == "cuda"
    for decision_batch in batches(decisions, batch_size, rng):
        batch = make_identity_batch(bundle, decision_batch, device)
        decision_weights = torch.from_numpy(
            weight_by_decision[decision_batch].copy()
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            loss, parts = weighted_loss(
                forward_identity(model, batch),
                batch,
                decision_weights,
                class_weights,
                opponent_loss_weight,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        count = len(decision_batch)
        examples += count
        for key, value in parts.items():
            totals[key] += value * count
    elapsed = time.perf_counter() - started
    return {
        **{key: float(value / max(examples, 1)) for key, value in totals.items()},
        "decisions": examples,
        "seconds": elapsed,
        "decisionsPerSecond": examples / max(elapsed, 1e-9),
    }


def _cycle_batch(
    decisions: np.ndarray,
    order: np.ndarray,
    cursor: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    if cursor + batch_size <= len(order):
        chosen = order[cursor : cursor + batch_size]
        return decisions[chosen], order, cursor + batch_size
    first = order[cursor:]
    order = rng.permutation(len(decisions))
    need = batch_size - len(first)
    chosen = np.concatenate((first, order[:need]))
    return decisions[chosen], order, need


def train_balanced_multideck_epoch(
    model: PTCGDeckIdentityTransformerPolicy,
    sources: list[tuple[IdentityBundle, np.ndarray, np.ndarray]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    batch_size_per_deck: int,
    rng: np.random.Generator,
    class_weights: torch.Tensor,
    opponent_loss_weight: float,
) -> dict[str, Any]:
    model.train()
    amp = device.type == "cuda"
    orders = [rng.permutation(len(decisions)) for _, decisions, _ in sources]
    cursors = [0 for _ in sources]
    weight_maps: list[np.ndarray] = []
    for bundle, decisions, weights in sources:
        values = np.zeros(
            len(bundle.sequence.base.data["episode_ids"]), dtype=np.float32
        )
        values[decisions] = weights
        weight_maps.append(values)
    steps = max(math.ceil(len(decisions) / batch_size_per_deck) for _, decisions, _ in sources)
    source_totals = {bundle.name: Counter() for bundle, _, _ in sources}
    source_examples = Counter()
    started = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for index, (bundle, decisions, _) in enumerate(sources):
            decision_batch, orders[index], cursors[index] = _cycle_batch(
                decisions,
                orders[index],
                cursors[index],
                min(batch_size_per_deck, len(decisions)),
                rng,
            )
            batch = make_identity_batch(bundle, decision_batch, device)
            decision_weights = torch.from_numpy(
                weight_maps[index][decision_batch].copy()
            ).to(device)
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                loss, parts = weighted_loss(
                    forward_identity(model, batch),
                    batch,
                    decision_weights,
                    class_weights,
                    opponent_loss_weight,
                )
            losses.append(loss)
            count = len(decision_batch)
            source_examples[bundle.name] += count
            for key, value in parts.items():
                source_totals[bundle.name][key] += value * count
        loss = torch.stack(losses).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    elapsed = time.perf_counter() - started
    return {
        "steps": steps,
        "seconds": elapsed,
        "sourceMetrics": {
            name: {
                **{
                    key: float(value / max(source_examples[name], 1))
                    for key, value in totals.items()
                },
                "decisions": int(source_examples[name]),
            }
            for name, totals in source_totals.items()
        },
    }


def _balanced_accuracy(targets: list[int], predictions: list[int]) -> float | None:
    if not targets:
        return None
    per_class = []
    for value in sorted(set(targets)):
        indices = [index for index, target in enumerate(targets) if target == value]
        per_class.append(
            sum(predictions[index] == value for index in indices) / len(indices)
        )
    return float(np.mean(per_class))


@torch.inference_mode()
def evaluate_identity(
    model: PTCGDeckIdentityTransformerPolicy,
    bundle: IdentityBundle,
    decisions: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    exact_index: list[int] = []
    exact_semantic: list[int] = []
    exact_semantic_main: list[int] = []
    top1_semantic: list[int] = []
    top3_semantic: list[int] = []
    count_correct: list[int] = []
    value_predictions: list[float] = []
    value_targets: list[float] = []
    class_targets: list[int] = []
    class_predictions: list[int] = []
    class_confidences: list[float] = []
    episode_best: dict[tuple[int, int], tuple[int, int, int, float]] = {}
    errors = 0
    for decision_batch in batches(decisions, batch_size, None):
        batch = make_identity_batch(bundle, decision_batch, device)
        option_logits, count_logits, value_logits, opponent_logits = forward_identity(
            model, batch
        )
        count_predictions = masked_count_logits(
            count_logits, batch["min_count"], batch["max_count"]
        ).argmax(dim=1).cpu().numpy()
        option_scores = option_logits.float().cpu().numpy()
        probabilities = torch.softmax(opponent_logits.float(), dim=1).cpu().numpy()
        for row, decision_value in enumerate(decision_batch):
            decision = int(decision_value)
            begin = int(bundle.sequence.base.data["option_offsets"][decision])
            end = int(bundle.sequence.base.data["option_offsets"][decision + 1])
            count = end - begin
            predicted_count = int(count_predictions[row])
            if not (
                int(bundle.sequence.base.data["min_counts"][decision])
                <= predicted_count
                <= int(bundle.sequence.base.data["max_counts"][decision])
            ):
                errors += 1
                predicted_count = int(bundle.sequence.base.data["min_counts"][decision])
            ranking = np.argsort(-option_scores[row, :count], kind="stable")
            predicted = [int(value) for value in ranking[:predicted_count]]
            expert = [
                int(value)
                for value in np.flatnonzero(
                    bundle.sequence.base.data["option_labels"][begin:end]
                )
            ]
            exact_index.append(int(set(predicted) == set(expert)))
            hashes = np.asarray(
                bundle.sequence.base.semantic_hash[decision, :count], dtype=np.uint32
            )
            predicted_semantic = Counter(int(hashes[index]) for index in predicted)
            expert_semantic = Counter(int(hashes[index]) for index in expert)
            semantic_match = int(predicted_semantic == expert_semantic)
            exact_semantic.append(semantic_match)
            if int(bundle.sequence.base.data["select_contexts"][decision]) == 0:
                exact_semantic_main.append(semantic_match)
            count_correct.append(int(predicted_count == len(expert)))
            if len(expert) == 1 and count > 1:
                expert_hash = int(hashes[expert[0]])
                top1_semantic.append(int(int(hashes[ranking[0]]) == expert_hash))
                top3_semantic.append(
                    int(expert_hash in {int(hashes[index]) for index in ranking[:3]})
                )

            label = int(bundle.opponent_labels[decision])
            visible = int(bundle.opponent_visible_unique[decision])
            if label >= 0 and visible > 0:
                prediction = int(np.argmax(probabilities[row]))
                confidence = float(np.max(probabilities[row]))
                class_targets.append(label)
                class_predictions.append(prediction)
                class_confidences.append(confidence)
                key = (
                    int(bundle.sequence.base.data["episode_ids"][decision]),
                    int(bundle.sequence.base.data["player_indices"][decision]),
                )
                candidate = (visible, label, prediction, confidence)
                if key not in episode_best or visible >= episode_best[key][0]:
                    episode_best[key] = candidate
        value_predictions.extend(torch.sigmoid(value_logits).float().cpu().numpy().tolist())
        value_targets.extend(batch["winner"].float().cpu().numpy().tolist())

    values = np.asarray(value_predictions, dtype=np.float64)
    targets = np.asarray(value_targets, dtype=np.float64)
    majority = Counter(class_targets).most_common(1)[0][1] / len(class_targets) if class_targets else None
    high = [index for index, value in enumerate(class_confidences) if value >= 0.60]
    episode_targets = [value[1] for value in episode_best.values()]
    episode_predictions = [value[2] for value in episode_best.values()]
    episode_majority = (
        Counter(episode_targets).most_common(1)[0][1] / len(episode_targets)
        if episode_targets
        else None
    )
    return {
        "decisions": int(len(decisions)),
        "exactIndex": float(np.mean(exact_index)) if exact_index else None,
        "exactSemantic": float(np.mean(exact_semantic)) if exact_semantic else None,
        "exactSemanticMain": float(np.mean(exact_semantic_main)) if exact_semantic_main else None,
        "singleChoiceTop1Semantic": float(np.mean(top1_semantic)) if top1_semantic else None,
        "singleChoiceTop3Semantic": float(np.mean(top3_semantic)) if top3_semantic else None,
        "countAccuracy": float(np.mean(count_correct)) if count_correct else None,
        "valueBrier": float(np.mean((values - targets) ** 2)) if len(values) else None,
        "illegalPredictionCount": errors,
        "opponentDeck": {
            "decisionExamples": len(class_targets),
            "decisionAccuracy": (
                float(np.mean(np.equal(class_targets, class_predictions)))
                if class_targets
                else None
            ),
            "decisionBalancedAccuracy": _balanced_accuracy(class_targets, class_predictions),
            "decisionMajorityAccuracy": majority,
            "confidence60Coverage": len(high) / len(class_targets) if class_targets else None,
            "confidence60Accuracy": (
                float(np.mean([class_targets[index] == class_predictions[index] for index in high]))
                if high
                else None
            ),
            "actorEpisodes": len(episode_targets),
            "actorEpisodeAccuracy": (
                float(np.mean(np.equal(episode_targets, episode_predictions)))
                if episode_targets
                else None
            ),
            "actorEpisodeBalancedAccuracy": _balanced_accuracy(
                episode_targets, episode_predictions
            ),
            "actorEpisodeMajorityAccuracy": episode_majority,
        },
    }


def save_checkpoint(
    path: Path, model: PTCGDeckIdentityTransformerPolicy, metadata: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config.to_dict(),
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "metadata": metadata,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-features", type=Path, required=True)
    parser.add_argument("--pretrain-cache", type=Path, required=True)
    parser.add_argument("--pretrain-sequence-cache", type=Path, required=True)
    parser.add_argument("--pretrain-identity-cache", type=Path, required=True)
    parser.add_argument("--pretrain-catalog", type=Path, required=True)
    parser.add_argument(
        "--current-source",
        nargs=6,
        action="append",
        metavar=("NAME", "FEATURES", "TOKEN_CACHE", "SEQUENCE_CACHE", "IDENTITY_CACHE", "CAL_EPISODES"),
        required=True,
    )
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--pretrain-batch", type=int, default=128)
    parser.add_argument("--finetune-batch-per-deck", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=384)
    parser.add_argument("--card-vocab", type=int, default=1600)
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--opponent-loss-weight", type=float, default=0.05)
    parser.add_argument("--tiny-decisions", type=int, default=0)
    parser.add_argument("--evaluate-holdout", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_payload = json.loads(args.class_map.read_text(encoding="utf-8"))
    opponent_classes = len(class_payload["classes"])
    pretrain = IdentityBundle.load(
        "pretrain",
        args.pretrain_features,
        args.pretrain_cache,
        args.pretrain_sequence_cache,
        args.pretrain_identity_cache,
    )
    currents: list[tuple[IdentityBundle, int]] = []
    for name, features, token_cache, sequence_cache, identity_cache, calibration in args.current_source:
        currents.append(
            (
                IdentityBundle.load(
                    name,
                    Path(features),
                    Path(token_cache),
                    Path(sequence_cache),
                    Path(identity_cache),
                ),
                int(calibration),
            )
        )
    if len(currents) < 2:
        raise RuntimeError("multi-deck training requires at least two current sources")
    if len({bundle.sequence.history_length for bundle, _ in currents} | {pretrain.sequence.history_length}) != 1:
        raise RuntimeError("all history lengths must match")

    pretrain_decisions = np.flatnonzero(pretrain.sequence.base.nontrivial_mask())
    module_by_episode = load_module_by_episode(args.pretrain_catalog)
    pretrain_weights = np.asarray(
        [
            float(pretrain.sequence.base.data["policy_weights"][decision])
            * MODULE_WEIGHTS.get(
                module_by_episode[int(pretrain.sequence.base.data["episode_ids"][decision])], 1.0
            )
            for decision in pretrain_decisions
        ],
        dtype=np.float32,
    )
    split_data: dict[str, dict[str, Any]] = {}
    fit_sources: list[tuple[IdentityBundle, np.ndarray, np.ndarray]] = []
    for bundle, calibration_count in currents:
        base = bundle.sequence.base
        meaningful = base.nontrivial_mask()
        holdout = np.flatnonzero(base.data["validation"] == 1)
        train = np.flatnonzero((base.data["validation"] == 0) & meaningful)
        train_episodes = episode_order(base, train)
        if calibration_count <= 0 or calibration_count >= len(train_episodes):
            raise RuntimeError(f"{bundle.name}: invalid calibration episode count")
        calibration_episodes = set(train_episodes[-calibration_count:])
        fit_episodes = set(train_episodes[:-calibration_count])
        fit = decisions_for_episodes(base, fit_episodes, meaningful)
        calibration = decisions_for_episodes(base, calibration_episodes, meaningful)
        weights = base.data["policy_weights"][fit].astype(np.float32)
        fit_sources.append((bundle, fit, weights))
        split_data[bundle.name] = {
            "fit": fit,
            "calibration": calibration,
            "holdout": holdout,
            "fitEpisodes": len(fit_episodes),
            "calibrationEpisodes": len(calibration_episodes),
            "holdoutEpisodes": len(set(int(value) for value in base.data["episode_ids"][holdout])),
        }

    class_counts = np.zeros(opponent_classes, dtype=np.int64)
    for bundle, decisions, _ in fit_sources:
        labels = np.asarray(bundle.opponent_labels[decisions], dtype=np.int64)
        visible = np.asarray(bundle.opponent_visible_unique[decisions], dtype=np.int64)
        valid = (labels >= 0) & (visible > 0)
        class_counts += np.bincount(labels[valid], minlength=opponent_classes)
    positive_classes = class_counts > 0
    if not np.any(positive_classes):
        raise RuntimeError("fit split has no visible opponent-class examples")
    class_weights_np = np.zeros(opponent_classes, dtype=np.float64)
    class_weights_np[positive_classes] = 1.0 / np.sqrt(
        class_counts[positive_classes].astype(np.float64)
    )
    class_weights_np[positive_classes] /= class_weights_np[positive_classes].mean()
    class_weights = torch.from_numpy(class_weights_np.astype(np.float32)).to(device)

    if args.tiny_decisions:
        take = min(args.tiny_decisions, len(pretrain_decisions))
        pretrain_decisions = pretrain_decisions[:take]
        pretrain_weights = pretrain_weights[:take]
        fit_sources = [
            (bundle, decisions[: min(take, len(decisions))], weights[: min(take, len(weights))])
            for bundle, decisions, weights in fit_sources
        ]

    config = DeckIdentityModelConfig(
        card_vocab=args.card_vocab,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        ff_dim=args.ff_dim,
        history_length=pretrain.sequence.history_length,
        opponent_classes=opponent_classes,
    )
    model = PTCGDeckIdentityTransformerPolicy(config).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "experiment": "exact own-deck multiset token plus visible-only opponent-deck auxiliary classifier",
        "device": str(device),
        "torchVersion": torch.__version__,
        "seed": args.seed,
        "privacyBoundary": (
            "own deck is submission-visible; opponent token uses actor-visible entities only; "
            "catalog deck hashes are training labels and never runtime inputs"
        ),
        "selectionBoundary": "per-deck chronological training tail calibration; holdouts sealed unless explicitly requested",
        "modelConfig": config.to_dict(),
        "parameterCount": model.parameter_count,
        "classMap": class_payload,
        "classFitCounts": class_counts.tolist(),
        "classWeights": class_weights_np.tolist(),
        "opponentLossWeight": args.opponent_loss_weight,
        "splits": {
            name: {
                "fitDecisions": int(len(values["fit"])),
                "calibrationDecisions": int(len(values["calibration"])),
                "holdoutDecisions": int(len(values["holdout"])),
                "fitEpisodes": values["fitEpisodes"],
                "calibrationEpisodes": values["calibrationEpisodes"],
                "holdoutEpisodes": values["holdoutEpisodes"],
            }
            for name, values in split_data.items()
        },
        "pretrainDecisions": int(len(pretrain_decisions)),
        "pretrain": [],
        "finetune": [],
    }
    print(
        json.dumps(
            {
                "stage": "start",
                "device": str(device),
                "parameterCount": model.parameter_count,
                "splits": report["splits"],
                "classFitCounts": class_counts.tolist(),
            }
        ),
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=1e-4)
    for epoch in range(1, args.pretrain_epochs + 1):
        metrics = train_single_epoch(
            model,
            pretrain,
            pretrain_decisions,
            pretrain_weights,
            optimizer,
            scaler,
            device,
            args.pretrain_batch,
            rng,
            class_weights,
            0.0,
        )
        metrics["epoch"] = epoch
        report["pretrain"].append(metrics)
        print(json.dumps({"stage": "pretrain", **metrics}), flush=True)

    baseline_model: PTCGSequenceTransformerPolicy | None = None
    if args.baseline_checkpoint:
        checkpoint = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
        baseline_model = PTCGSequenceTransformerPolicy(
            SequenceModelConfig(**checkpoint["config"])
        ).to(device)
        baseline_model.load_state_dict(checkpoint["state_dict"])
        report["baselineCalibration"] = {
            bundle.name: evaluate_sequence(
                baseline_model,
                bundle.sequence,
                split_data[bundle.name]["calibration"],
                device,
                args.finetune_batch_per_deck,
            )
            for bundle, _ in currents
        }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=1e-4)
    best_score = -1.0
    best_epoch = 0
    best_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.finetune_epochs + 1):
        train_metrics = train_balanced_multideck_epoch(
            model,
            fit_sources,
            optimizer,
            scaler,
            device,
            args.finetune_batch_per_deck,
            rng,
            class_weights,
            args.opponent_loss_weight,
        )
        calibration = {
            bundle.name: evaluate_identity(
                model,
                bundle,
                split_data[bundle.name]["calibration"],
                device,
                args.finetune_batch_per_deck,
            )
            for bundle, _ in currents
        }
        macro = float(np.mean([value["exactSemantic"] for value in calibration.values()]))
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "calibration": calibration,
            "calibrationMacroExactSemantic": macro,
        }
        report["finetune"].append(row)
        print(json.dumps({"stage": "finetune", **row}), flush=True)
        if macro > best_score:
            best_score = macro
            best_epoch = epoch
            save_checkpoint(
                best_path,
                model,
                {"selectedFineTuneEpoch": epoch, "calibration": calibration, "macro": macro},
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    report["selectedFineTuneEpoch"] = best_epoch
    report["selectedCalibrationMacroExactSemantic"] = best_score
    report["selectedCalibration"] = {
        bundle.name: evaluate_identity(
            model,
            bundle,
            split_data[bundle.name]["calibration"],
            device,
            args.finetune_batch_per_deck,
        )
        for bundle, _ in currents
    }
    report["holdout"] = (
        {
            bundle.name: evaluate_identity(
                model,
                bundle,
                split_data[bundle.name]["holdout"],
                device,
                args.finetune_batch_per_deck,
            )
            for bundle, _ in currents
        }
        if args.evaluate_holdout
        else None
    )
    if args.evaluate_holdout and baseline_model is not None:
        report["baselineHoldout"] = {
            bundle.name: evaluate_sequence(
                baseline_model,
                bundle.sequence,
                split_data[bundle.name]["holdout"],
                device,
                args.finetune_batch_per_deck,
            )
            for bundle, _ in currents
        }
    report_path = args.output_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "selectedEpoch": best_epoch,
                "selectedCalibrationMacroExactSemantic": best_score,
                "holdoutOpened": bool(args.evaluate_holdout),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
