from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from static_deck_bc_common import (
    STRICT_PREDICATE,
    assert_specialist_receipt,
    load_json,
    matching_archetypes,
    read_deck_card_names,
    validation_episode,
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_filter(path: Path):
    spec = importlib.util.spec_from_file_location("strict_scoregt1000_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import strict builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.filter_day


def rows_and_seats(
    catalog_path: Path,
    deck_names: dict[str, tuple[str, ...]],
    config: dict[str, Any],
    archetype: str,
) -> tuple[list[dict[str, str]], dict[int, dict[str, Any]], dict[str, int]]:
    threshold = float(config["minScoreExclusive"])
    selected: list[dict[str, str]] = []
    lookup: dict[int, dict[str, Any]] = {}
    excluded = Counter()
    seen: set[int] = set()
    with catalog_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            episode = int(row["episode_id"])
            if episode in seen:
                excluded["duplicateEpisode"] += 1
                continue
            seen.add(episode)
            if row.get("is_clean") != "1":
                excluded["notClean"] += 1
                continue
            score = float(row["min_score"])
            if score <= threshold:
                excluded["scoreLE1000"] += 1
                continue
            seats = []
            ambiguous = False
            for seat in (0, 1):
                deck_hash = str(row[f"deck{seat}_sha256"])
                matches = matching_archetypes(deck_names.get(deck_hash, ()), config)
                if len(matches) > 1:
                    ambiguous = True
                elif matches == [archetype]:
                    seats.append(seat)
            if ambiguous:
                excluded["ambiguousDeck"] += 1
                continue
            if not seats:
                excluded["nonTarget"] += 1
                continue
            selected.append(row)
            lookup[episode] = {
                "targetSeats": seats,
                "winnerIndex": int(row["winner_index"]),
                "score": score,
                "deckHashes": [str(row[f"deck{seat}_sha256"]) for seat in seats],
            }
    return selected, lookup, dict(sorted(excluded.items()))


def write_catalog(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized_hash(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii")
    return str(value)


def rewrite_labels(
    output: Path,
    episode_lookup: dict[int, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    features = output / "features_tensordict"
    episode_ids = np.load(features / "episode_ids.npy", mmap_mode="r", allow_pickle=False)
    player_indices = np.load(features / "player_indices.npy", mmap_mode="r", allow_pickle=False)
    deck_hashes = np.load(features / "deck_hashes.npy", mmap_mode="r", allow_pickle=False)
    policy = np.load(features / "policy_weights.npy", mmap_mode="r+", allow_pickle=False)
    validation = np.load(features / "validation.npy", mmap_mode="r+", allow_pickle=False)
    winner_weight = float(config["policyWeights"]["targetWinner"])
    loss_weight = float(config["policyWeights"]["targetLossOrDraw"])
    policy[:] = 0
    validation[:] = 0
    weight_counts = Counter()
    train_episodes: set[int] = set()
    validation_episodes: set[int] = set()
    for index in range(len(episode_ids)):
        episode = int(episode_ids[index])
        if episode not in episode_lookup:
            raise ValueError(f"decision references unselected episode {episode}")
        split_validation = validation_episode(str(episode), config)
        validation[index] = int(split_validation)
        (validation_episodes if split_validation else train_episodes).add(episode)
        seat = int(player_indices[index])
        info = episode_lookup[episode]
        if seat not in info["targetSeats"]:
            weight_counts["nonTargetSeatDecisions"] += 1
            continue
        expected_hash = {info["deckHashes"][info["targetSeats"].index(seat)]}
        if normalized_hash(deck_hashes[index]) not in expected_hash:
            raise ValueError(f"seat/deck hash mismatch for episode {episode} seat {seat}")
        if int(info["winnerIndex"]) == seat:
            policy[index] = winner_weight
            weight_counts["winnerPolicyDecisions"] += 1
        else:
            policy[index] = loss_weight
            weight_counts["lossOrDrawPolicyDecisions"] += 1
    policy.flush()
    validation.flush()
    if train_episodes & validation_episodes:
        raise ValueError("stable split produced train/validation episode overlap")
    return {
        **dict(weight_counts),
        "decisions": int(len(episode_ids)),
        "policyDecisions": int(np.count_nonzero(np.asarray(policy) > 0)),
        "trainEpisodes": len(train_episodes),
        "validationEpisodes": len(validation_episodes),
        "trainValidationOverlap": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--archetype", required=True)
    parser.add_argument("--strict-day-root", type=Path, required=True)
    parser.add_argument("--source-catalog-dir", type=Path, required=True)
    parser.add_argument("--engine-catalog", type=Path, required=True)
    parser.add_argument("--strict-builder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rows", type=int, default=12)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    valid_ids = {row["id"] for row in config["archetypes"]}
    if args.archetype not in valid_ids:
        parser.error(f"unknown archetype: {args.archetype}")
    strict_day = args.strict_day_root.resolve()
    strict_receipt_path = strict_day / "audit-receipt.json"
    strict_receipt = load_json(strict_receipt_path)
    if not strict_receipt.get("parity", {}).get("passed"):
        raise ValueError("strict day has no passing parity receipt")
    strict_catalog = strict_day / "catalog" / "replay_catalog.csv"
    deck_names = read_deck_card_names(args.source_catalog_dir.resolve(), args.engine_catalog.resolve())
    # Count exclusions against the original broad day catalog, while still
    # insisting every selected row passes the strict per-game predicate.  The
    # tensor source below is the separately parity-checked strict day.
    selected, lookup, excluded = rows_and_seats(
        args.source_catalog_dir.resolve() / "replay_catalog.csv",
        deck_names,
        config,
        args.archetype,
    )
    if not selected:
        raise ValueError(f"no strict episodes for archetype {args.archetype}")
    with strict_catalog.open("r", encoding="utf-8", newline="") as handle:
        fields = list(csv.DictReader(handle).fieldnames or [])
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(f".{output.name}.{os.getpid()}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    source_layout = staging.parent / f".{output.name}.{os.getpid()}.source-layout"
    fake_root = source_layout / "prepared" / "universal"
    fake_root.mkdir(parents=True)
    write_catalog(source_layout / "prepared" / "catalog" / "replay_catalog.csv", selected, fields)
    dataset = {
        "name": strict_receipt["day"],
        "root": str(fake_root),
        "features": str(strict_day / "features_tensordict"),
        "decisions": str(strict_day / "decisions.jsonl.gz"),
        "tokenCache": str(strict_day / "token_cache"),
        "sequenceCache": str(strict_day / "sequence_cache"),
        "identityCache": str(strict_day / "identity_cache"),
        "summary": strict_receipt["summary"],
    }
    try:
        filter_day = load_filter(args.strict_builder.resolve())
        result = filter_day(
            dataset,
            staging,
            threshold=float(config["minScoreExclusive"]),
            sample_rows=args.sample_rows,
        )
        label_receipt = rewrite_labels(staging, lookup, config)
        scores = [float(row["min_score"]) for row in selected]
        episode_ids = [int(row["episode_id"]) for row in selected]
        receipt = {
            "schemaVersion": 1,
            "kind": "experiment7_static_deck_bc_day_shard",
            "profile": f"{config['profilePrefix']}:{args.archetype}",
            "archetype": args.archetype,
            "day": strict_receipt["day"],
            "strictPredicate": STRICT_PREDICATE,
            "minScoreExclusive": float(config["minScoreExclusive"]),
            "sourceStrictDay": str(strict_day),
            "sourceStrictDayReceipt": str(strict_receipt_path),
            "sourceStrictDayReceiptSha256": sha256(strict_receipt_path),
            "episodes": len(episode_ids),
            "duplicateEpisodes": len(episode_ids) - len(set(episode_ids)),
            "decisions": label_receipt["decisions"],
            "policyDecisions": label_receipt["policyDecisions"],
            "scoreMin": min(scores),
            "scoreMax": max(scores),
            "excludedCounts": excluded,
            "policyWeights": config["policyWeights"],
            "split": config["stableSplit"],
            "parityReceipt": "audit-receipt.json",
            "parityPassed": True,
            **label_receipt,
        }
        assert_specialist_receipt(receipt)
        atomic_json(staging / "specialist-receipt.json", receipt)
        atomic_json(staging / "SUCCESS", {
            "status": "SUCCESS",
            "receipt": "specialist-receipt.json",
            "receiptSha256": sha256(staging / "specialist-receipt.json"),
        })
        os.replace(staging, output)
        print(json.dumps({**receipt, "output": str(output)}, ensure_ascii=False))
    finally:
        if source_layout.exists():
            shutil.rmtree(source_layout)
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
