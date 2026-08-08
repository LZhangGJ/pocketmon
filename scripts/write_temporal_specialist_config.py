from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPORAL_ARCHITECTURE = (
    "structured_card_attack_deepsets_deck_transformer8_masked_pointer_with_stop"
)
STRUCTURED_ARCHITECTURE = (
    "structured_card_attack_deepsets_deck_masked_pointer_with_stop"
)


def resolve(value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_architecture(path: Path) -> str:
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("config", {}).get(
        "architecture",
        STRUCTURED_ARCHITECTURE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the immutable RL-BC-004 Transformer-8 experiment config"
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--deck-map", type=Path, required=True)
    parser.add_argument("--target-deck", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--card-database", type=Path, required=True)
    parser.add_argument("--attack-database", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=20260720)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--base-learning-rate", type=float, default=1e-4)
    parser.add_argument("--temporal-learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ffn-dim", type=int, default=768)
    parser.add_argument("--transformer-dropout", type=float, default=0.10)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.epochs != 12 or args.history_length != 8:
        raise ValueError("RL-BC-004 is pre-registered for 12 epochs and 8 history steps")

    input_path = resolve(args.input)
    deck_map = resolve(args.deck_map)
    target_deck = resolve(args.target_deck)
    initialize_from = resolve(args.initialize_from)
    card_database = resolve(args.card_database)
    attack_database = resolve(args.attack_database)
    output = resolve(args.output)
    for path in (
        input_path,
        deck_map,
        target_deck,
        initialize_from,
        card_database,
        attack_database,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite planned config: {output}")

    source_architecture = load_checkpoint_architecture(initialize_from)
    if source_architecture not in (STRUCTURED_ARCHITECTURE, TEMPORAL_ARCHITECTURE):
        raise ValueError(
            f"unsupported warm-start architecture: {source_architecture}"
        )

    config = {
        "experiment_id": args.experiment_id,
        "arm": "fixed-deck-structured-transformer8-warm-start",
        "architecture": TEMPORAL_ARCHITECTURE,
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "dataset": {
            "deck_map_sha256": sha256(deck_map),
            "target_deck_sha256": sha256(target_deck),
        },
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
            "base_learning_rate": args.base_learning_rate,
            "temporal_learning_rate": args.temporal_learning_rate,
            "hidden_dim": args.hidden_dim,
            "early_stopping_patience": args.patience,
            "value_loss_weight": 0.25,
            "gradient_clip_norm": 1.0,
        },
        "history": {
            "enabled": True,
            "encoder": "transformer",
            "max_length": args.history_length,
            "group_by": ["episode_id", "player"],
            "order_by": "action_step",
            "token": "prior pre-action state plus that prior selected-option summary",
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
            "card_database_sha256": sha256(card_database),
            "attack_database_sha256": sha256(attack_database),
            "confidence_threshold": args.confidence_threshold,
        },
        "warm_start": {
            "checkpoint_sha256": sha256(initialize_from),
            "source_architecture": source_architecture,
        },
        "policy_loss_rows": "policy_weight == 1 only",
        "value_loss_rows": "value_weight == 1 for both players",
        "initialization": "warm_start",
        "offline_rl": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
