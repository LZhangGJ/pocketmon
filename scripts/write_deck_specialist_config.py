from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a reproducible warm-start specialist training config")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--deck-map", required=True)
    parser.add_argument("--card-database", required=True)
    parser.add_argument("--attack-database", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=20260720)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = resolve(args.input)
    deck_map = resolve(args.deck_map)
    card_database = resolve(args.card_database)
    attack_database = resolve(args.attack_database)
    output = resolve(args.output)
    config = {
        "experiment_id": args.experiment_id,
        "arm": "deck-specialist-warm-start",
        "architecture": "structured_card_attack_deepsets_deck_masked_pointer_with_stop",
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "split": {
            "kind": "episode_id",
            "seed": args.split_seed,
            "train_fraction": 0.8,
            "validation_fraction": 0.2,
        },
        "training": {
            "formal_seeds": [args.seed],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_dim": args.hidden_dim,
            "early_stopping_patience": args.patience,
            "value_loss_weight": 0.25,
            "gradient_clip_norm": 1.0,
        },
        "history": {"enabled": False, "encoder": "none", "max_length": 0},
        "structured": {
            "card_attack_embeddings": True,
            "entity_encoder": "deepsets_masked_mean_max",
            "deck_conditioning": "acting_player_submitted_deck_masked_mean",
            "deck_map_sha256": sha256(deck_map),
            "card_database_sha256": sha256(card_database),
            "attack_database_sha256": sha256(attack_database),
            "confidence_threshold": args.confidence_threshold,
        },
        "policy_loss_rows": "policy_weight == 1 only",
        "value_loss_rows": "value_weight == 1 for both players",
        "random_initialization": False,
        "initialization": "warm_start",
        "offline_rl": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
