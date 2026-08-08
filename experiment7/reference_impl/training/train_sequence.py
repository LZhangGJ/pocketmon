from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sequence_model import PTCGSequenceTransformerPolicy, SequenceModelConfig
from public_event_model import PublicEventModelConfig, PTCGPublicEventTransformerPolicy
from deck_knowledge_model import DeckKnowledgeModelConfig, PTCGDeckKnowledgeTransformerPolicy
from train import (
    MODULE_WEIGHTS,
    Bundle,
    batches,
    decisions_for_episodes,
    episode_order,
    load_module_by_episode,
    loss_terms,
    make_batch,
    masked_count_logits,
)


@dataclass
class SequenceBundle:
    base: Bundle
    history_indices: np.ndarray
    expert_action_features: np.ndarray
    opponent_event_cat: np.ndarray | None = None
    opponent_event_num: np.ndarray | None = None
    opponent_event_mask: np.ndarray | None = None
    deck_features: np.ndarray | None = None
    deck_card_ids: np.ndarray | None = None

    @classmethod
    def load(
        cls,
        features_path: Path,
        token_cache: Path,
        sequence_cache: Path,
        public_event_cache: Path | None = None,
        deck_knowledge_cache: Path | None = None,
    ) -> "SequenceBundle":
        base = Bundle.load(features_path, token_cache)
        history = np.load(sequence_cache / "history_indices.npy", mmap_mode="r")
        actions = np.load(sequence_cache / "expert_action_features.npy", mmap_mode="r")
        decisions = len(base.data["episode_ids"])
        if history.shape[0] != decisions or actions.shape != (
            decisions,
            base.data["option_features"].shape[1],
        ):
            raise RuntimeError(f"sequence cache shape mismatch for {features_path}")
        event_cat = event_num = event_mask = None
        if public_event_cache is not None:
            event_cat = np.load(public_event_cache / "opponent_event_cat.npy", mmap_mode="r")
            event_num = np.load(public_event_cache / "opponent_event_num.npy", mmap_mode="r")
            event_mask = np.load(public_event_cache / "opponent_event_mask.npy", mmap_mode="r")
            if (
                event_cat.shape[0] != decisions
                or event_num.shape[:2] != event_cat.shape[:2]
                or event_mask.shape != event_cat.shape[:2]
            ):
                raise RuntimeError(f"public event cache shape mismatch for {features_path}")
        deck_features = deck_card_ids = None
        if deck_knowledge_cache is not None:
            deck_features = np.load(deck_knowledge_cache / "deck_features.npy", mmap_mode="r")
            deck_card_ids = np.load(deck_knowledge_cache / "deck_card_ids.npy")
            if deck_features.shape[0] != decisions or deck_features.shape[1] != len(deck_card_ids):
                raise RuntimeError(f"deck knowledge cache shape mismatch for {features_path}")
        return cls(
            base=base,
            history_indices=history,
            expert_action_features=actions,
            opponent_event_cat=event_cat,
            opponent_event_num=event_num,
            opponent_event_mask=event_mask,
            deck_features=deck_features,
            deck_card_ids=deck_card_ids,
        )

    @property
    def history_length(self) -> int:
        return int(self.history_indices.shape[1])


def make_sequence_batch(
    bundle: SequenceBundle, decisions: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    result = make_batch(bundle.base, decisions, device)
    history_indices = np.asarray(bundle.history_indices[decisions], dtype=np.int64).copy()
    history_mask = history_indices >= 0
    safe = np.maximum(history_indices, 0)
    history_state = bundle.base.data["state_features"][safe].astype(np.float32, copy=True)
    history_action = np.asarray(
        bundle.expert_action_features[safe], dtype=np.float32
    ).copy()
    history_state[~history_mask] = 0.0
    history_action[~history_mask] = 0.0
    result["history_state"] = torch.from_numpy(history_state).to(device)
    result["history_action"] = torch.from_numpy(history_action).to(device)
    result["history_mask"] = torch.from_numpy(history_mask.astype(np.uint8)).to(device)
    if bundle.opponent_event_cat is not None:
        result["opponent_event_cat"] = torch.from_numpy(
            np.asarray(bundle.opponent_event_cat[decisions], dtype=np.int64).copy()
        ).to(device)
        result["opponent_event_num"] = torch.from_numpy(
            np.asarray(bundle.opponent_event_num[decisions], dtype=np.float32).copy()
        ).to(device)
        result["opponent_event_mask"] = torch.from_numpy(
            np.asarray(bundle.opponent_event_mask[decisions], dtype=np.uint8).copy()
        ).to(device)
    if bundle.deck_features is not None:
        result["deck_features"] = torch.from_numpy(
            np.asarray(bundle.deck_features[decisions], dtype=np.float32).copy()
        ).to(device)
    return result


def forward_batch(
    model: PTCGSequenceTransformerPolicy | PTCGPublicEventTransformerPolicy | PTCGDeckKnowledgeTransformerPolicy,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if "opponent_event_cat" in batch:
        return model(
            batch["state"],
            batch["history_state"],
            batch["history_action"],
            batch["history_mask"],
            batch["opponent_event_cat"],
            batch["opponent_event_num"],
            batch["opponent_event_mask"],
            batch["entity_cat"],
            batch["entity_num"],
            batch["entity_mask"],
            batch["options"],
            batch["option_mask"],
        )
    if "deck_features" in batch:
        return model(
            batch["state"],
            batch["history_state"],
            batch["history_action"],
            batch["history_mask"],
            batch["deck_features"],
            batch["entity_cat"],
            batch["entity_num"],
            batch["entity_mask"],
            batch["options"],
            batch["option_mask"],
        )
    return model(
        batch["state"],
        batch["history_state"],
        batch["history_action"],
        batch["history_mask"],
        batch["entity_cat"],
        batch["entity_num"],
        batch["entity_mask"],
        batch["options"],
        batch["option_mask"],
    )


def train_epoch(
    model: PTCGSequenceTransformerPolicy,
    bundle: SequenceBundle,
    decisions: np.ndarray,
    weights: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    model.train()
    weight_by_decision = np.zeros(len(bundle.base.data["episode_ids"]), dtype=np.float32)
    weight_by_decision[decisions] = weights
    totals = Counter()
    examples = 0
    started = time.perf_counter()
    amp = device.type == "cuda"
    for decision_batch in batches(decisions, batch_size, rng):
        batch = make_sequence_batch(bundle, decision_batch, device)
        decision_weights = torch.from_numpy(weight_by_decision[decision_batch].copy()).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp
        ):
            outputs = forward_batch(model, batch)
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
    model: PTCGSequenceTransformerPolicy,
    bundle: SequenceBundle,
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
    errors = 0
    for decision_batch in batches(decisions, batch_size, None):
        batch = make_sequence_batch(bundle, decision_batch, device)
        option_logits, count_logits, value_logits = forward_batch(model, batch)
        count_predictions = masked_count_logits(
            count_logits, batch["min_count"], batch["max_count"]
        ).argmax(dim=1).cpu().numpy()
        option_scores = option_logits.float().cpu().numpy()
        for row, decision in enumerate(decision_batch):
            begin = int(bundle.base.data["option_offsets"][decision])
            end = int(bundle.base.data["option_offsets"][decision + 1])
            count = end - begin
            predicted_count = int(count_predictions[row])
            if not (
                int(bundle.base.data["min_counts"][decision])
                <= predicted_count
                <= int(bundle.base.data["max_counts"][decision])
            ):
                errors += 1
                predicted_count = int(bundle.base.data["min_counts"][decision])
            ranking = np.argsort(-option_scores[row, :count], kind="stable")
            predicted = [int(value) for value in ranking[:predicted_count]]
            expert = [
                int(value)
                for value in np.flatnonzero(
                    bundle.base.data["option_labels"][begin:end]
                )
            ]
            exact_index.append(int(set(predicted) == set(expert)))
            hashes = np.asarray(bundle.base.semantic_hash[decision, :count], dtype=np.uint32)
            predicted_semantic = Counter(int(hashes[index]) for index in predicted)
            expert_semantic = Counter(int(hashes[index]) for index in expert)
            semantic_match = int(predicted_semantic == expert_semantic)
            exact_semantic.append(semantic_match)
            if int(bundle.base.data["select_contexts"][decision]) == 0:
                exact_semantic_main.append(semantic_match)
            count_correct.append(int(predicted_count == len(expert)))
            if len(expert) == 1 and count > 1:
                expert_hash = int(hashes[expert[0]])
                top1_semantic.append(int(int(hashes[ranking[0]]) == expert_hash))
                top3_semantic.append(
                    int(expert_hash in {int(hashes[index]) for index in ranking[:3]})
                )
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


def save_checkpoint(
    path: Path, model: PTCGSequenceTransformerPolicy, metadata: dict[str, Any]
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
    parser.add_argument("--pretrain-public-event-cache", type=Path)
    parser.add_argument("--pretrain-deck-knowledge-cache", type=Path)
    parser.add_argument("--pretrain-catalog", type=Path, required=True)
    parser.add_argument("--current-features", type=Path, required=True)
    parser.add_argument("--current-cache", type=Path, required=True)
    parser.add_argument("--current-sequence-cache", type=Path, required=True)
    parser.add_argument("--current-public-event-cache", type=Path)
    parser.add_argument("--current-deck-knowledge-cache", type=Path)
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
    parser.add_argument(
        "--evaluate-holdout",
        action="store_true",
        help="Open the chronological holdout only for a final frozen candidate.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_public_events = (
        args.pretrain_public_event_cache is not None
        and args.current_public_event_cache is not None
    )
    if (args.pretrain_public_event_cache is None) != (args.current_public_event_cache is None):
        raise RuntimeError("both public event caches must be supplied together")
    use_deck_knowledge = (
        args.pretrain_deck_knowledge_cache is not None
        and args.current_deck_knowledge_cache is not None
    )
    if (args.pretrain_deck_knowledge_cache is None) != (args.current_deck_knowledge_cache is None):
        raise RuntimeError("both deck knowledge caches must be supplied together")
    if use_public_events and use_deck_knowledge:
        raise RuntimeError("experiment isolation forbids combining public events and deck knowledge")
    pretrain = SequenceBundle.load(
        args.pretrain_features,
        args.pretrain_cache,
        args.pretrain_sequence_cache,
        args.pretrain_public_event_cache,
        args.pretrain_deck_knowledge_cache,
    )
    current = SequenceBundle.load(
        args.current_features,
        args.current_cache,
        args.current_sequence_cache,
        args.current_public_event_cache,
        args.current_deck_knowledge_cache,
    )
    if pretrain.history_length != current.history_length:
        raise RuntimeError("pretrain and current history length differ")
    pretrain_decisions = np.flatnonzero(pretrain.base.nontrivial_mask())
    current_holdout = np.flatnonzero(current.base.data["validation"] == 1)
    current_train = np.flatnonzero(
        (current.base.data["validation"] == 0) & current.base.nontrivial_mask()
    )
    train_episodes = episode_order(current.base, current_train)
    if args.calibration_episodes >= len(train_episodes):
        raise RuntimeError("calibration split leaves no fine-tune episodes")
    calibration_episodes = set(train_episodes[-args.calibration_episodes :])
    fit_episodes = set(train_episodes[: -args.calibration_episodes])
    finetune_decisions = decisions_for_episodes(
        current.base, fit_episodes, current.base.nontrivial_mask()
    )
    calibration_decisions = decisions_for_episodes(
        current.base, calibration_episodes, current.base.nontrivial_mask()
    )

    module_by_episode = load_module_by_episode(args.pretrain_catalog)
    pretrain_weights = np.asarray(
        [
            float(pretrain.base.data["policy_weights"][decision])
            * MODULE_WEIGHTS.get(
                module_by_episode[int(pretrain.base.data["episode_ids"][decision])], 1.0
            )
            for decision in pretrain_decisions
        ],
        dtype=np.float32,
    )
    finetune_weights = current.base.data["policy_weights"][finetune_decisions].astype(
        np.float32
    )
    if args.tiny_decisions:
        take = min(args.tiny_decisions, len(pretrain_decisions))
        pretrain_decisions = pretrain_decisions[:take]
        pretrain_weights = pretrain_weights[:take]
        finetune_decisions = finetune_decisions[: min(take, len(finetune_decisions))]
        finetune_weights = finetune_weights[: len(finetune_decisions)]

    if use_public_events:
        assert pretrain.opponent_event_cat is not None
        config = PublicEventModelConfig(
            card_vocab=args.card_vocab,
            attack_vocab=args.card_vocab,
            d_model=args.d_model,
            n_heads=args.heads,
            n_layers=args.layers,
            ff_dim=args.ff_dim,
            history_length=pretrain.history_length,
            opponent_event_length=int(pretrain.opponent_event_cat.shape[1]),
        )
        model = PTCGPublicEventTransformerPolicy(config).to(device)
    elif use_deck_knowledge:
        assert pretrain.deck_features is not None and pretrain.deck_card_ids is not None
        assert current.deck_features is not None and current.deck_card_ids is not None
        if not np.array_equal(pretrain.deck_card_ids, current.deck_card_ids):
            raise RuntimeError("pretrain/current deck card ID order differs")
        config = DeckKnowledgeModelConfig(
            card_vocab=args.card_vocab,
            d_model=args.d_model,
            n_heads=args.heads,
            n_layers=args.layers,
            ff_dim=args.ff_dim,
            history_length=pretrain.history_length,
            deck_card_types=int(pretrain.deck_features.shape[1]),
            deck_num_dim=int(pretrain.deck_features.shape[2]),
        )
        model = PTCGDeckKnowledgeTransformerPolicy(config).to(device)
    else:
        config = SequenceModelConfig(
            card_vocab=args.card_vocab,
            d_model=args.d_model,
            n_heads=args.heads,
            n_layers=args.layers,
            ff_dim=args.ff_dim,
            history_length=pretrain.history_length,
        )
        model = PTCGSequenceTransformerPolicy(config).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schemaVersion": 2,
        "device": str(device),
        "torchVersion": torch.__version__,
        "seed": args.seed,
        "historyBoundary": "same episodeId and playerIndex; strictly prior sourceStep only",
        "publicOpponentEvents": use_public_events,
        "deckKnowledge": use_deck_knowledge,
        "deckKnowledgeBoundary": (
            "own fixed deck plus actor-visible zones; exact full-deck views when available; otherwise bounds/probabilities"
            if use_deck_knowledge
            else None
        ),
        "publicEventPrivacyBoundary": (
            "acting-player observation.logs only; opponent playerIndex only; serial fields excluded"
            if use_public_events
            else None
        ),
        "modelConfig": config.to_dict(),
        "parameterCount": model.parameter_count,
        "splits": {
            "pretrainDecisions": int(len(pretrain_decisions)),
            "fineTuneDecisions": int(len(finetune_decisions)),
            "calibrationDecisions": int(len(calibration_decisions)),
            "holdoutDecisions": int(len(current_holdout)),
            "fineTuneEpisodes": len(fit_episodes),
            "calibrationEpisodes": len(calibration_episodes),
            "holdoutEpisodes": len(
                set(
                    int(value)
                    for value in current.base.data["episode_ids"][current_holdout]
                )
            ),
        },
        "pretrain": [],
        "finetune": [],
    }
    print(
        json.dumps(
            {
                "stage": "start",
                **{
                    key: report[key]
                    for key in ("device", "parameterCount", "modelConfig", "splits")
                },
            }
        ),
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.pretrain_lr, weight_decay=1e-4
    )
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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.finetune_lr, weight_decay=1e-4
    )
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
        calibration = evaluate(
            model, current, calibration_decisions, device, args.finetune_batch
        )
        score = float(calibration["exactSemantic"] or 0.0)
        row = {"epoch": epoch, "train": train_metrics, "calibration": calibration}
        report["finetune"].append(row)
        print(json.dumps({"stage": "finetune", **row}), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_checkpoint(
                best_path,
                model,
                {"selectedFineTuneEpoch": epoch, "calibration": calibration},
            )

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    report["selectedFineTuneEpoch"] = best_epoch
    report["selectedCalibration"] = evaluate(
        model, current, calibration_decisions, device, args.finetune_batch
    )
    report["holdout"] = (
        evaluate(model, current, current_holdout, device, args.finetune_batch)
        if args.evaluate_holdout
        else None
    )
    report_path = args.output_dir / "training_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "stage": "complete",
                "selectedEpoch": best_epoch,
                "holdout": report["holdout"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
