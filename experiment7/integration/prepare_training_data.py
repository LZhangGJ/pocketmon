from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_from_pocketmon_replays import build_dataset
from build_replay_catalog import build_catalog
from common import (
    DatasetPaths,
    Experiment7Error,
    parse_version,
    read_csv,
    resolve_python,
    run_checked,
    sha256_file,
    utc_now,
    write_json,
)
from select_initial_decks import choose_decks


def build_engine_catalog(cards_path: Path, attacks_path: Path, output: Path) -> dict[str, Any]:
    cards_payload = json.loads(cards_path.read_text(encoding="utf-8"))
    attacks_payload = json.loads(attacks_path.read_text(encoding="utf-8"))
    cards = cards_payload if isinstance(cards_payload, list) else cards_payload["cards"]
    attacks = attacks_payload if isinstance(attacks_payload, list) else attacks_payload["attacks"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"cards": cards, "attacks": attacks}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        "path": str(output.resolve()),
        "sha256": sha256_file(output),
        "cards": len(cards),
        "attacks": len(attacks),
        "cardVocab": max(int(row["cardId"]) for row in cards) + 1,
    }


def choose_module_window(catalog: Path, current_override: str | None, pretrain_override: str | None) -> tuple[str, list[str], dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"episodes": 0, "latest": 0.0, "policyDecisions": 0}
    )
    for row in read_csv(catalog):
        if row.get("is_clean") != "1" or not row.get("module_version"):
            continue
        module = row["module_version"]
        stats[module]["episodes"] += 1
        stats[module]["latest"] = max(stats[module]["latest"], float(row.get("create_timestamp", 0.0) or 0.0))
        stats[module]["policyDecisions"] += int(row.get("policy_decisions0", 0) or 0) + int(row.get("policy_decisions1", 0) or 0)
    if not stats:
        raise Experiment7Error("catalog has no clean module versions")
    current = current_override or max(
        stats,
        key=lambda module: (
            float(stats[module]["latest"]),
            int(stats[module]["episodes"]),
            parse_version(module),
        ),
    )
    if current not in stats:
        raise Experiment7Error(f"requested current module {current!r} not found in clean catalog")
    ordered = sorted(stats, key=parse_version)
    if pretrain_override:
        pretrain = [value.strip() for value in pretrain_override.split(",") if value.strip()]
        missing = sorted(set(pretrain) - set(stats))
        if missing:
            raise Experiment7Error(f"pretrain modules not found: {missing}")
    else:
        position = ordered.index(current)
        pretrain = ordered[max(0, position - 1) : position + 1]
    return current, pretrain, {module: dict(values) for module, values in stats.items()}


def vendor_command(
    python: str,
    reference_root: Path,
    script: str,
    arguments: list[str],
    log: Path,
) -> None:
    pipeline = reference_root / "data_pipeline"
    run_checked(
        [python, str(pipeline / script), *arguments],
        cwd=pipeline,
        env={"PYTHONPATH": str(pipeline)},
        log_path=log,
    )


def build_standard_caches(
    *,
    python: str,
    reference_root: Path,
    engine_catalog: Path,
    dataset_root: Path,
    log_root: Path,
) -> tuple[Path, Path]:
    token_cache = dataset_root / "token_cache"
    sequence_cache = dataset_root / "sequence_cache"
    vendor_command(
        python,
        reference_root,
        "build_token_cache.py",
        [
            "--decisions",
            str(dataset_root / "decisions.jsonl.gz"),
            "--features",
            str(dataset_root / "features.npz"),
            "--catalog",
            str(engine_catalog),
            "--output-dir",
            str(token_cache),
        ],
        log_root / f"{dataset_root.name}_token_cache.log",
    )
    vendor_command(
        python,
        reference_root,
        "build_sequence_cache.py",
        [
            "--features",
            str(dataset_root / "features.npz"),
            "--output-dir",
            str(sequence_cache),
            "--history-length",
            "8",
        ],
        log_root / f"{dataset_root.name}_sequence_cache.log",
    )
    return token_cache, sequence_cache


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    python = resolve_python(args.python)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    reference_root = args.reference_root.resolve()
    for path in [
        reference_root / "data_pipeline" / "features.py",
        reference_root / "training" / "train_multideck_identity.py",
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    engine_catalog = output_root / "engine_catalog.json"
    engine_receipt = build_engine_catalog(args.cards.resolve(), args.attacks.resolve(), engine_catalog)
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
    current_module, pretrain_modules, module_stats = choose_module_window(
        catalog, args.current_module, args.pretrain_modules
    )

    selection_root = output_root / "selection"
    selection = choose_decks(
        args.ladder_dir.resolve(),
        catalog,
        selection_root,
        args.target_deck.resolve(),
        args.desired_decks,
        args.minimum_decks,
        args.min_actor_episodes,
        args.min_policy_decisions,
        args.near_duplicate_threshold,
        "Mega Lucario",
        current_module,
    )
    selected = selection["selected"]

    datasets_root = output_root / "datasets"
    pretrain_root = datasets_root / "pretrain"
    build_dataset(
        reference_root=reference_root,
        catalog=catalog,
        engine_catalog=engine_catalog,
        output_dir=pretrain_root,
        mode="broad",
        deck_sha256=None,
        module_versions=",".join(pretrain_modules),
        validation_fraction=0.0,
        positive_policy_only=True,
        skip_forced=True,
        max_episodes=args.max_pretrain_episodes,
    )
    pretrain_token, pretrain_sequence = build_standard_caches(
        python=python,
        reference_root=reference_root,
        engine_catalog=engine_catalog,
        dataset_root=pretrain_root,
        log_root=logs,
    )

    current_paths: list[DatasetPaths] = []
    for candidate in selected:
        name = candidate["name"]
        root = datasets_root / name
        summary = build_dataset(
            reference_root=reference_root,
            catalog=catalog,
            engine_catalog=engine_catalog,
            output_dir=root,
            mode="exact",
            deck_sha256=candidate["deckSha256"],
            module_versions=current_module,
            validation_fraction=args.holdout_fraction,
            positive_policy_only=True,
            skip_forced=True,
            max_episodes=args.max_current_episodes,
        )
        train_episodes = int(summary["trainEpisodes"])
        calibration = max(1, int(math.ceil(train_episodes * args.calibration_fraction)))
        if calibration >= train_episodes:
            raise Experiment7Error(
                f"{name}: {train_episodes} train episodes leave no fit set after {calibration} calibration episodes"
            )
        token_cache, sequence_cache = build_standard_caches(
            python=python,
            reference_root=reference_root,
            engine_catalog=engine_catalog,
            dataset_root=root,
            log_root=logs,
        )
        current_paths.append(
            DatasetPaths(
                name=name,
                root=root,
                features=root / "features.npz",
                decisions=root / "decisions.jsonl.gz",
                catalog=catalog,
                token_cache=token_cache,
                sequence_cache=sequence_cache,
                identity_cache=root / "identity_cache",
                calibration_episodes=calibration,
                deck_sha256=candidate["deckSha256"],
                deck_path=Path(candidate["deckPath"]),
            )
        )

    class_map = output_root / "opponent_class_map.json"
    source_arguments: list[str] = []
    for source in current_paths:
        source_arguments.extend(
            [
                "--source",
                source.name,
                str(source.decisions),
                str(catalog),
                str(source.calibration_episodes),
            ]
        )
    vendor_command(
        python,
        reference_root,
        "build_opponent_deck_class_map.py",
        [
            *source_arguments,
            "--min-train-actors",
            str(args.min_opponent_class_actors),
            "--output",
            str(class_map),
        ],
        logs / "opponent_class_map.log",
    )

    all_identity_sources = [
        DatasetPaths(
            name="pretrain",
            root=pretrain_root,
            features=pretrain_root / "features.npz",
            decisions=pretrain_root / "decisions.jsonl.gz",
            catalog=catalog,
            token_cache=pretrain_token,
            sequence_cache=pretrain_sequence,
            identity_cache=pretrain_root / "identity_cache",
            calibration_episodes=0,
        ),
        *current_paths,
    ]
    for source in all_identity_sources:
        vendor_command(
            python,
            reference_root,
            "build_deck_identity_cache.py",
            [
                "--decisions",
                str(source.decisions),
                "--features",
                str(source.features),
                "--token-cache",
                str(source.token_cache),
                "--catalog",
                str(catalog),
                "--class-map",
                str(class_map),
                "--deck-map",
                str(deck_map),
                "--output-dir",
                str(source.identity_cache),
            ],
            logs / f"{source.name}_identity_cache.log",
        )

    manifest = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "referenceRoot": str(reference_root),
        "engineCatalog": engine_receipt,
        "catalog": {"path": str(catalog), "sha256": sha256_file(catalog)},
        "deckMap": {"path": str(deck_map), "sha256": sha256_file(deck_map)},
        "classMap": {"path": str(class_map), "sha256": sha256_file(class_map)},
        "moduleSelection": {
            "current": current_module,
            "pretrain": pretrain_modules,
            "catalogStats": module_stats,
        },
        "pretrain": {
            "name": "pretrain",
            "root": str(pretrain_root),
            "features": str(pretrain_root / "features.npz"),
            "decisions": str(pretrain_root / "decisions.jsonl.gz"),
            "tokenCache": str(pretrain_token),
            "sequenceCache": str(pretrain_sequence),
            "identityCache": str(pretrain_root / "identity_cache"),
        },
        "currentSources": [
            {
                "name": source.name,
                "root": str(source.root),
                "features": str(source.features),
                "decisions": str(source.decisions),
                "tokenCache": str(source.token_cache),
                "sequenceCache": str(source.sequence_cache),
                "identityCache": str(source.identity_cache),
                "calibrationEpisodes": source.calibration_episodes,
                "deckSha256": source.deck_sha256,
                "deckPath": str(source.deck_path),
            }
            for source in current_paths
        ],
        "selectionReceipt": str(selection_root / "selected_decks.json"),
        "privacyBoundary": "all raw decisions are sanitized before feature and token construction",
        "holdoutOpened": False,
    }
    manifest_path = output_root / "training_sources.json"
    write_json(manifest_path, manifest)
    payload = {
        "trainingSources": str(manifest_path),
        "selectedDecks": len(current_paths),
        "currentModule": current_module,
        "pretrainModules": pretrain_modules,
        "cardVocab": engine_receipt["cardVocab"],
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare all caches for Experiment 7 multi-deck training")
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--ladder-dir", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--attacks", type=Path, required=True)
    parser.add_argument("--target-deck", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python")
    parser.add_argument("--policy-source", choices=("winners", "nonlosers", "both"), default="winners")
    parser.add_argument("--current-module")
    parser.add_argument("--pretrain-modules")
    parser.add_argument("--desired-decks", type=int, default=6)
    parser.add_argument("--minimum-decks", type=int, default=4)
    parser.add_argument("--min-actor-episodes", type=int, default=10)
    parser.add_argument("--min-policy-decisions", type=int, default=500)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.80)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--min-opponent-class-actors", type=int, default=3)
    parser.add_argument("--strict-catalog", action="store_true")
    parser.add_argument("--max-replay-files", type=int, default=0)
    parser.add_argument("--max-pretrain-episodes", type=int, default=0)
    parser.add_argument("--max-current-episodes", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.minimum_decks <= args.desired_decks:
        raise ValueError("require 1 <= minimum_decks <= desired_decks")
    prepare(args)


if __name__ == "__main__":
    main()
