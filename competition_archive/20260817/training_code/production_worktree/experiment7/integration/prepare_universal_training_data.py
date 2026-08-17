from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_from_pocketmon_replays import build_dataset
from build_replay_catalog import build_catalog
from common import resolve_python, sha256_file, utc_now, write_json
from prepare_training_data import build_engine_catalog, build_standard_caches, vendor_command


def reuse_engine_catalog(source: Path, output: Path) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    cards = payload.get("cards")
    attacks = payload.get("attacks")
    if not isinstance(cards, list) or not cards or not isinstance(attacks, list) or not attacks:
        raise ValueError(f"invalid combined engine catalog: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"cards": cards, "attacks": attacks}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "sourcePath": str(source.resolve()),
        "sourceSha256": sha256_file(source),
        "cards": len(cards),
        "attacks": len(attacks),
        "cardVocab": max(int(row["cardId"]) for row in cards) + 1,
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Build one chronological, all-deck, both-player Experiment 7 dataset."""
    python = resolve_python(args.python)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    reference_root = args.reference_root.resolve()

    engine_catalog = output_root / "engine_catalog.json"
    if args.engine_catalog:
        engine_receipt = reuse_engine_catalog(args.engine_catalog.resolve(), engine_catalog)
    else:
        if args.cards is None or args.attacks is None:
            raise ValueError("provide --engine-catalog or both --cards and --attacks")
        engine_receipt = build_engine_catalog(
            args.cards.resolve(), args.attacks.resolve(), engine_catalog
        )
    catalog_root = output_root / "catalog"
    build_catalog(
        args.raw_root.resolve(),
        catalog_root,
        args.policy_source,
        args.strict_catalog,
        args.max_replay_files,
    )
    catalog = catalog_root / "replay_catalog.csv"
    deck_map = catalog_root / "deck_map.json"

    dataset_root = output_root / "universal"
    summary = build_dataset(
        reference_root=reference_root,
        catalog=catalog,
        engine_catalog=engine_catalog,
        output_dir=dataset_root,
        mode="broad",
        deck_sha256=None,
        module_versions=args.module_versions,
        validation_fraction=args.validation_fraction,
        positive_policy_only=False,
        skip_forced=True,
        max_episodes=args.max_episodes,
        min_game_score_exclusive=args.min_game_score_exclusive,
    )
    token_cache, sequence_cache = build_standard_caches(
        python=python,
        reference_root=reference_root,
        engine_catalog=engine_catalog,
        dataset_root=dataset_root,
        log_root=logs,
    )

    # The exact-opponent classifier was diagnostic-only in the replay audit.
    # Retain a one-class cache contract so own-deck card loading stays identical,
    # while Universal BC trains with opponent auxiliary weight zero.
    class_map = output_root / "opponent_class_map.json"
    write_json(
        class_map,
        {
            "schemaVersion": 1,
            "classes": [{"index": 0, "name": "OTHER"}],
            "other": {"deckHashes": []},
            "purpose": "own-deck identity cache only; opponent auxiliary disabled",
        },
    )
    identity_cache = dataset_root / "identity_cache"
    vendor_command(
        python,
        reference_root,
        "build_deck_identity_cache.py",
        [
            "--decisions",
            str(dataset_root / "decisions.jsonl.gz"),
            "--features",
            str(dataset_root / "features.npz"),
            "--token-cache",
            str(token_cache),
            "--catalog",
            str(catalog),
            "--class-map",
            str(class_map),
            "--deck-map",
            str(deck_map),
            "--output-dir",
            str(identity_cache),
        ],
        logs / "universal_identity_cache.log",
    )

    manifest = {
        "schemaVersion": 2,
        "kind": "experiment7_universal_bc",
        "createdAt": utc_now(),
        "referenceRoot": str(reference_root),
        "engineCatalog": engine_receipt,
        "catalog": {"path": str(catalog), "sha256": sha256_file(catalog)},
        "deckMap": {"path": str(deck_map), "sha256": sha256_file(deck_map)},
        "classMap": {"path": str(class_map), "sha256": sha256_file(class_map)},
        "moduleVersions": args.module_versions,
        "policySource": args.policy_source,
        "minGameScoreExclusive": args.min_game_score_exclusive,
        "dataset": {
            "name": "universal",
            "root": str(dataset_root),
            "features": str(dataset_root / "features.npz"),
            "decisions": str(dataset_root / "decisions.jsonl.gz"),
            "tokenCache": str(token_cache),
            "sequenceCache": str(sequence_cache),
            "identityCache": str(identity_cache),
            "summary": summary,
        },
        "trainingContract": {
            "policy": "positive replay policy weights only (winner by default)",
            "value": "both player perspectives, independent unit weights",
            "opponentDeckAuxiliary": "disabled",
            "validation": "chronological newest replay episodes",
            "scoreFilter": (
                f"manifest min_score > {args.min_game_score_exclusive}"
                if args.min_game_score_exclusive is not None
                else "none"
            ),
        },
        "privacyBoundary": "all raw observations sanitized before cache construction",
    }
    manifest_path = output_root / "universal_training_sources.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "trainingSources": str(manifest_path),
                "episodes": summary["sourceEpisodes"],
                "decisions": summary["decisions"],
                "winnerDecisions": summary["winnerDecisions"],
                "cardVocab": engine_receipt["cardVocab"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare all-history, all-deck Universal BC caches"
    )
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--engine-catalog",
        type=Path,
        help="Reuse a verified combined engine catalog from an earlier Experiment 7 run",
    )
    parser.add_argument("--cards", type=Path)
    parser.add_argument("--attacks", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python")
    parser.add_argument(
        "--policy-source", choices=("winners", "nonlosers", "both"), default="winners"
    )
    parser.add_argument(
        "--module-versions",
        default="*",
        help="Comma-separated compatible modules; '*' uses every clean catalog module",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--strict-catalog", action="store_true")
    parser.add_argument("--max-replay-files", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument(
        "--min-game-score-exclusive",
        type=float,
        help="Keep only games where both players' score floor is strictly above this value",
    )
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
