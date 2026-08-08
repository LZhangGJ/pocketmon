from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.nn import functional as F

from model import ModelConfig, PTCGTransformerPolicy


MODULE_WEIGHTS = {"1.32.2": 0.25, "1.32.3": 0.50, "1.32.4": 1.00, "1.32.5": 1.00}


@dataclass
class Bundle:
    features_path: Path
    cache_dir: Path
    data: dict[str, np.ndarray]
    entity_cat: np.ndarray
    entity_num: np.ndarray
    entity_mask: np.ndarray
    semantic_hash: np.ndarray
    semantic_labels: np.ndarray

    @classmethod
    def load(cls, features_path: Path, cache_dir: Path) -> "Bundle":
        with np.load(features_path) as archive:
            data = {name: archive[name] for name in archive.files}
        result = cls(
            features_path=features_path,
            cache_dir=cache_dir,
            data=data,
            entity_cat=np.load(cache_dir / "entity_cat.npy", mmap_mode="r"),
            entity_num=np.load(cache_dir / "entity_num.npy", mmap_mode="r"),
            entity_mask=np.load(cache_dir / "entity_mask.npy", mmap_mode="r"),
            semantic_hash=np.load(cache_dir / "semantic_hash.npy", mmap_mode="r"),
            semantic_labels=np.load(cache_dir / "semantic_labels.npy", mmap_mode="r"),
        )
        decisions = len(data["episode_ids"])
        if any(array.shape[0] != decisions for array in (
            result.entity_cat,
            result.entity_num,
            result.entity_mask,
            result.semantic_hash,
            result.semantic_labels,
        )):
            raise RuntimeError(f"cache decision count mismatch for {features_path}")
        return result

    def option_counts(self) -> np.ndarray:
        return self.data["option_offsets"][1:] - self.data["option_offsets"][:-1]

    def nontrivial_mask(self) -> np.ndarray:
        counts = self.option_counts()
        forced = (self.data["min_counts"] == self.data["max_counts"]) & (
            (self.data["min_counts"] == 0) | (self.data["min_counts"] == counts)
        )
        return ~forced


def load_module_by_episode(path: Path) -> dict[int, str]:
    result: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            episode = int(row["episode_id"])
            module = row["module_version"]
            previous = result.setdefault(episode, module)
            if previous != module:
                raise RuntimeError(f"episode {episode} has conflicting modules")
    return result


def episode_order(bundle: Bundle, decisions: np.ndarray) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for decision in decisions:
        episode = int(bundle.data["episode_ids"][decision])
        if episode not in seen:
            seen.add(episode)
            result.append(episode)
    return result


def decisions_for_episodes(bundle: Bundle, episodes: set[int], meaningful: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(bundle.data["episode_ids"], list(episodes)) & meaningful)


def batches(decisions: np.ndarray, batch_size: int, rng: np.random.Generator | None) -> Iterable[np.ndarray]:
    order = np.arange(len(decisions)) if rng is None else rng.permutation(len(decisions))
    for begin in range(0, len(order), batch_size):
        yield decisions[order[begin : begin + batch_size]]


def make_batch(bundle: Bundle, decisions: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    counts = bundle.option_counts()[decisions].astype(np.int32)
    max_actions = int(counts.max(initial=1))
    entity_counts = bundle.entity_mask[decisions].sum(axis=1).astype(np.int32)
    max_entities = int(entity_counts.max(initial=1))
    batch_size = len(decisions)

    options = np.zeros((batch_size, max_actions, bundle.data["option_features"].shape[1]), dtype=np.float32)
    option_mask = np.zeros((batch_size, max_actions), dtype=np.uint8)
    original_labels = np.zeros((batch_size, max_actions), dtype=np.float32)
    for row, decision in enumerate(decisions):
        begin = int(bundle.data["option_offsets"][decision])
        end = int(bundle.data["option_offsets"][decision + 1])
        count = end - begin
        options[row, :count] = bundle.data["option_features"][begin:end]
        option_mask[row, :count] = 1
        original_labels[row, :count] = bundle.data["option_labels"][begin:end]

    semantic_labels = np.asarray(bundle.semantic_labels[decisions, :max_actions], dtype=np.float32).copy()
    result = {
        "decision": torch.from_numpy(decisions.astype(np.int64, copy=True)).to(device),
        "state": torch.from_numpy(bundle.data["state_features"][decisions].astype(np.float32, copy=True)).to(device),
        "entity_cat": torch.from_numpy(
            np.asarray(bundle.entity_cat[decisions, :max_entities], dtype=np.int64).copy()
        ).to(device),
        "entity_num": torch.from_numpy(
            np.asarray(bundle.entity_num[decisions, :max_entities], dtype=np.float32).copy()
        ).to(device),
        "entity_mask": torch.from_numpy(
            np.asarray(bundle.entity_mask[decisions, :max_entities], dtype=np.uint8).copy()
        ).to(device),
        "options": torch.from_numpy(options).to(device),
        "option_mask": torch.from_numpy(option_mask).to(device),
        "semantic_labels": torch.from_numpy(semantic_labels).to(device),
        "original_labels": torch.from_numpy(original_labels).to(device),
        "min_count": torch.from_numpy(bundle.data["min_counts"][decisions].astype(np.int64, copy=True)).to(device),
        "max_count": torch.from_numpy(bundle.data["max_counts"][decisions].astype(np.int64, copy=True)).to(device),
        "chosen_count": torch.from_numpy(bundle.data["chosen_counts"][decisions].astype(np.int64, copy=True)).to(device),
        "winner": torch.from_numpy(bundle.data["is_winners"][decisions].astype(np.float32, copy=True)).to(device),
    }
    return result


def masked_count_logits(logits: torch.Tensor, min_count: torch.Tensor, max_count: torch.Tensor) -> torch.Tensor:
    values = torch.arange(logits.shape[1], device=logits.device)[None, :]
    valid = (values >= min_count[:, None]) & (values <= max_count[:, None])
    return logits.masked_fill(~valid, -1e4)


def loss_terms(
    option_logits: torch.Tensor,
    count_logits: torch.Tensor,
    value_logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    decision_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["option_mask"].bool()
    labels = batch["semantic_labels"]
    chosen_count = batch["chosen_count"]

    all_lse = torch.logsumexp(option_logits.masked_fill(~mask, -1e4), dim=1)
    positive_lse = torch.logsumexp(option_logits.masked_fill(~(mask & labels.bool()), -1e4), dim=1)
    single_loss = all_lse - positive_lse

    raw_bce = F.binary_cross_entropy_with_logits(option_logits, labels, reduction="none")
    positives = (labels * mask).sum(dim=1, keepdim=True)
    negatives = ((1.0 - labels) * mask).sum(dim=1, keepdim=True)
    positive_weight = torch.where(positives > 0, 0.5 / positives.clamp_min(1.0), torch.zeros_like(positives))
    negative_weight = torch.where(
        negatives > 0,
        torch.where(positives > 0, 0.5 / negatives.clamp_min(1.0), 1.0 / negatives.clamp_min(1.0)),
        torch.zeros_like(negatives),
    )
    balance = torch.where(labels > 0, positive_weight, negative_weight) * mask
    multi_loss = (raw_bce * balance).sum(dim=1)
    policy_per_decision = torch.where(chosen_count == 1, single_loss, multi_loss)
    policy_loss = (policy_per_decision * decision_weights).sum() / decision_weights.sum().clamp_min(1e-6)

    count_masked = masked_count_logits(count_logits, batch["min_count"], batch["max_count"])
    count_per_decision = F.cross_entropy(count_masked, chosen_count, reduction="none")
    count_loss = (count_per_decision * decision_weights).sum() / decision_weights.sum().clamp_min(1e-6)
    value_per_decision = F.binary_cross_entropy_with_logits(value_logits, batch["winner"], reduction="none")
    value_loss = (value_per_decision * decision_weights).sum() / decision_weights.sum().clamp_min(1e-6)
    total = policy_loss + 0.20 * count_loss + 0.05 * value_loss
    return total, {
        "policy": float(policy_loss.detach().cpu()),
        "count": float(count_loss.detach().cpu()),
        "value": float(value_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def train_epoch(
    model: PTCGTransformerPolicy,
    bundle: Bundle,
    decisions: np.ndarray,
    weights: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    weight_by_decision = np.zeros(len(bundle.data["episode_ids"]), dtype=np.float32)
    weight_by_decision[decisions] = weights
    totals = Counter()
    examples = 0
    started = time.perf_counter()
    amp = device.type == "cuda"
    for decision_batch in batches(decisions, batch_size, rng):
        batch = make_batch(bundle, decision_batch, device)
        decision_weights = torch.from_numpy(weight_by_decision[decision_batch].copy()).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            outputs = model(
                batch["state"],
                batch["entity_cat"],
                batch["entity_num"],
                batch["entity_mask"],
                batch["options"],
                batch["option_mask"],
            )
            loss, parts = loss_terms(*outputs, batch, decision_weights)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        examples += len(decision_batch)
        for key, value in parts.items():
            totals[key] += value * len(decision_batch)
    elapsed = time.perf_counter() - started
    return {
        **{key: float(value / max(examples, 1)) for key, value in totals.items()},
        "decisions": examples,
        "seconds": elapsed,
        "decisionsPerSecond": examples / max(elapsed, 1e-9),
    }


@torch.inference_mode()
def evaluate(
    model: PTCGTransformerPolicy,
    bundle: Bundle,
    decisions: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    exact_index = []
    exact_semantic = []
    exact_semantic_main = []
    top1_semantic = []
    top3_semantic = []
    count_correct = []
    value_predictions = []
    value_targets = []
    errors = 0
    for decision_batch in batches(decisions, batch_size, None):
        batch = make_batch(bundle, decision_batch, device)
        option_logits, count_logits, value_logits = model(
            batch["state"],
            batch["entity_cat"],
            batch["entity_num"],
            batch["entity_mask"],
            batch["options"],
            batch["option_mask"],
        )
        count_predictions = masked_count_logits(
            count_logits, batch["min_count"], batch["max_count"]
        ).argmax(dim=1).cpu().numpy()
        option_scores = option_logits.float().cpu().numpy()
        for row, decision in enumerate(decision_batch):
            begin = int(bundle.data["option_offsets"][decision])
            end = int(bundle.data["option_offsets"][decision + 1])
            count = end - begin
            predicted_count = int(count_predictions[row])
            if not (int(bundle.data["min_counts"][decision]) <= predicted_count <= int(bundle.data["max_counts"][decision])):
                errors += 1
                predicted_count = int(bundle.data["min_counts"][decision])
            ranking = np.argsort(-option_scores[row, :count], kind="stable")
            predicted = [int(value) for value in ranking[:predicted_count]]
            expert = [int(value) for value in np.flatnonzero(bundle.data["option_labels"][begin:end])]
            exact_index.append(int(set(predicted) == set(expert)))
            hashes = np.asarray(bundle.semantic_hash[decision, :count], dtype=np.uint32)
            predicted_semantic = Counter(int(hashes[index]) for index in predicted)
            expert_semantic = Counter(int(hashes[index]) for index in expert)
            semantic_match = int(predicted_semantic == expert_semantic)
            exact_semantic.append(semantic_match)
            if int(bundle.data["select_contexts"][decision]) == 0:
                exact_semantic_main.append(semantic_match)
            count_correct.append(int(predicted_count == len(expert)))
            if len(expert) == 1 and count > 1:
                expert_hash = int(hashes[expert[0]])
                top1_semantic.append(int(int(hashes[ranking[0]]) == expert_hash))
                top3_semantic.append(int(expert_hash in set(int(hashes[index]) for index in ranking[:3])))
        value_predictions.extend(torch.sigmoid(value_logits).float().cpu().numpy().tolist())
        value_targets.extend(batch["winner"].float().cpu().numpy().tolist())
    values = np.asarray(value_predictions, dtype=np.float64)
    targets = np.asarray(value_targets, dtype=np.float64)
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
    }


def save_checkpoint(path: Path, model: PTCGTransformerPolicy, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": model.config.to_dict(),
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metadata": metadata,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-features", type=Path, required=True)
    parser.add_argument("--pretrain-cache", type=Path, required=True)
    parser.add_argument("--pretrain-catalog", type=Path, required=True)
    parser.add_argument("--current-features", type=Path, required=True)
    parser.add_argument("--current-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pretrain-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--pretrain-batch", type=int, default=128)
    parser.add_argument("--finetune-batch", type=int, default=64)
    parser.add_argument("--calibration-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=384)
    parser.add_argument("--card-vocab", type=int, default=1600)
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--tiny-decisions", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrain = Bundle.load(args.pretrain_features, args.pretrain_cache)
    current = Bundle.load(args.current_features, args.current_cache)
    pretrain_decisions = np.flatnonzero(pretrain.nontrivial_mask())
    current_holdout = np.flatnonzero(current.data["validation"] == 1)
    current_train = np.flatnonzero((current.data["validation"] == 0) & current.nontrivial_mask())
    train_episodes = episode_order(current, current_train)
    if args.calibration_episodes >= len(train_episodes):
        raise RuntimeError("calibration split leaves no fine-tune episodes")
    calibration_episodes = set(train_episodes[-args.calibration_episodes :])
    fit_episodes = set(train_episodes[: -args.calibration_episodes])
    finetune_decisions = decisions_for_episodes(current, fit_episodes, current.nontrivial_mask())
    calibration_decisions = decisions_for_episodes(
        current, calibration_episodes, current.nontrivial_mask()
    )

    module_by_episode = load_module_by_episode(args.pretrain_catalog)
    pretrain_weights = np.asarray(
        [
            float(pretrain.data["policy_weights"][decision])
            * MODULE_WEIGHTS[module_by_episode[int(pretrain.data["episode_ids"][decision])]]
            for decision in pretrain_decisions
        ],
        dtype=np.float32,
    )
    finetune_weights = current.data["policy_weights"][finetune_decisions].astype(np.float32)
    if args.tiny_decisions:
        take = min(args.tiny_decisions, len(pretrain_decisions))
        pretrain_decisions = pretrain_decisions[:take]
        pretrain_weights = pretrain_weights[:take]
        finetune_decisions = finetune_decisions[: min(take, len(finetune_decisions))]
        finetune_weights = finetune_weights[: len(finetune_decisions)]

    config = ModelConfig(
        card_vocab=args.card_vocab,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        ff_dim=args.ff_dim,
    )
    model = PTCGTransformerPolicy(config).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "device": str(device),
        "torchVersion": torch.__version__,
        "seed": args.seed,
        "modelConfig": config.to_dict(),
        "parameterCount": model.parameter_count,
        "splits": {
            "pretrainDecisions": int(len(pretrain_decisions)),
            "fineTuneDecisions": int(len(finetune_decisions)),
            "calibrationDecisions": int(len(calibration_decisions)),
            "holdoutDecisions": int(len(current_holdout)),
            "fineTuneEpisodes": len(fit_episodes),
            "calibrationEpisodes": len(calibration_episodes),
            "holdoutEpisodes": len(set(int(v) for v in current.data["episode_ids"][current_holdout])),
        },
        "pretrain": [],
        "finetune": [],
    }
    print(json.dumps({"stage": "start", **{k: report[k] for k in ("device", "parameterCount", "splits")}}), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pretrain_lr, weight_decay=1e-4)
    for epoch in range(1, args.pretrain_epochs + 1):
        metrics = train_epoch(
            model,
            pretrain,
            pretrain_decisions,
            pretrain_weights,
            optimizer,
            scaler,
            device,
            args.pretrain_batch,
            rng,
        )
        metrics["epoch"] = epoch
        report["pretrain"].append(metrics)
        print(json.dumps({"stage": "pretrain", **metrics}), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr, weight_decay=1e-4)
    best_score = -1.0
    best_epoch = 0
    best_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.finetune_epochs + 1):
        train_metrics = train_epoch(
            model,
            current,
            finetune_decisions,
            finetune_weights,
            optimizer,
            scaler,
            device,
            args.finetune_batch,
            rng,
        )
        calibration = evaluate(model, current, calibration_decisions, device, args.finetune_batch)
        score = float(calibration["exactSemantic"] or 0.0)
        row = {"epoch": epoch, "train": train_metrics, "calibration": calibration}
        report["finetune"].append(row)
        print(json.dumps({"stage": "finetune", **row}), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(best_path, model, {"selectedFineTuneEpoch": epoch, "calibration": calibration})

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    report["selectedFineTuneEpoch"] = best_epoch
    report["selectedCalibration"] = evaluate(
        model, current, calibration_decisions, device, args.finetune_batch
    )
    # The chronological holdout is opened once, after the epoch is frozen.
    report["holdout"] = evaluate(model, current, current_holdout, device, args.finetune_batch)
    report_path = args.output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"stage": "complete", "selectedEpoch": best_epoch, "holdout": report["holdout"]}), flush=True)


if __name__ == "__main__":
    main()
