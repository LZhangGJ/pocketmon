from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import train_universal_bc_async as async_train
from static_deck_bc_common import assert_specialist_receipt, load_json


def validate_static_sources(sources_path: Path, config: dict) -> dict:
    sources = load_json(sources_path)
    if sources.get("kind") != "experiment7_universal_bc":
        raise ValueError("static trainer requires Universal BC tensor sources")
    profile = sources.get("staticProfile")
    if not profile or not str(profile).startswith(config["profilePrefix"] + ":"):
        raise ValueError("source manifest lacks a static specialist profile")
    archetype = str(profile).split(":", 1)[1]
    seen: set[int] = set()
    train_episodes: set[int] = set()
    validation_episodes: set[int] = set()
    for dataset in sources.get("datasets", []):
        receipt_path = Path(dataset["specialistReceipt"])
        receipt = load_json(receipt_path)
        assert_specialist_receipt(receipt, float(config["minScoreExclusive"]))
        if receipt.get("archetype") != archetype:
            raise ValueError("mixed archetype source manifest")
        catalog = Path(dataset["root"]) / "catalog" / "replay_catalog.csv"
        catalog_episodes: set[int] = set()
        with catalog.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("is_clean") != "1" or float(row["min_score"]) <= float(config["minScoreExclusive"]):
                    raise ValueError("loader rejected non-strict source episode")
                episode = int(row["episode_id"])
                if episode in catalog_episodes or episode in seen:
                    raise ValueError("loader rejected duplicate source episode")
                catalog_episodes.add(episode)
        episode_ids = np.load(Path(dataset["features"]) / "episode_ids.npy", mmap_mode="r", allow_pickle=False)
        observed = {int(value) for value in np.unique(episode_ids)}
        if observed != catalog_episodes:
            raise ValueError("loader episode/catalog parity failure")
        validation = np.load(Path(dataset["features"]) / "validation.npy", mmap_mode="r", allow_pickle=False)
        train_episodes.update(int(value) for value in np.unique(episode_ids[validation == 0]))
        validation_episodes.update(int(value) for value in np.unique(episode_ids[validation == 1]))
        seen.update(catalog_episodes)
    if not seen:
        raise ValueError("static source manifest contains no episodes")
    if train_episodes & validation_episodes:
        raise ValueError("loader rejected train/validation episode overlap")
    return sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--resume-progress", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epoch-start", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.05)
    parser.add_argument("--prefetch-batches", type=int, default=6)
    parser.add_argument("--prefetch-workers", type=int, default=2)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    validate_static_sources(args.sources.resolve(), config)
    training = config["training"]
    forwarded = argparse.Namespace(
        sources=args.sources,
        output_dir=args.output_dir,
        initialize_from=args.initialize_from or Path(training["initializer"]),
        resume_progress=args.resume_progress,
        device=args.device,
        seed=20260815,
        epochs=int(training["maxEpochs"]),
        epoch_start=args.epoch_start,
        batch_size=args.batch_size or int(training["batchSize"]),
        prefetch_batches=args.prefetch_batches,
        prefetch_workers=args.prefetch_workers,
        learning_rate=args.learning_rate,
        learning_rate_schedule=[],
        weight_decay=args.weight_decay,
        value_loss_weight=args.value_loss_weight,
        validation_fraction=0.05,
        max_train_decisions=0,
        max_pending_validations=1,
        poll_seconds=5.0,
    )
    async_train.train(forwarded)


if __name__ == "__main__":
    main()
