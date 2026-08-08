from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "experiment7" / "reference_impl"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare all Experiment 7 multideck datasets and caches on one Linux coordinator"
    )
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-deck", type=Path, default=ROOT / "agents/lucario_rule/deck.csv")
    parser.add_argument("--cards", type=Path, default=ROOT / "data/reference/official_cards.json")
    parser.add_argument("--attacks", type=Path, default=ROOT / "data/reference/official_attacks.json")
    parser.add_argument("--desired-decks", type=int, default=6)
    parser.add_argument("--minimum-decks", type=int, default=4)
    parser.add_argument("--min-actor-episodes", type=int, default=40)
    parser.add_argument("--min-policy-decisions", type=int, default=800)
    parser.add_argument("--pretrain-max-decisions", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    required = [args.replay_dir, args.analysis_dir, args.target_deck, args.cards, args.attacks]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if not (args.analysis_dir / "archetype_summary.csv").is_file():
        raise FileNotFoundError(args.analysis_dir / "archetype_summary.csv")
    if not (args.analysis_dir / "representative_decklists.csv").is_file():
        raise FileNotFoundError(args.analysis_dir / "representative_decklists.csv")

    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"refusing to overwrite non-empty output root: {output}; pass --force")
    output.mkdir(parents=True, exist_ok=True)
    canonical_dir = output / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical = canonical_dir / "public_replay.jsonl.gz"
    conversion_report = canonical_dir / "conversion_report.json"
    sidecar = canonical_dir / "replay_decks.jsonl.gz"
    sidecar_report = canonical_dir / "deck_map_report.json"
    engine_catalog = output / "engine_catalog.json"
    selection_dir = output / "selection"
    dataset_root = output / "datasets"

    run(
        [
            sys.executable,
            str(ROOT / "scripts/convert_public_replays.py"),
            "--input-root", str(args.replay_dir.parent),
            "--date", args.replay_dir.name,
            "--alignment", "previous",
            "--policy-source", "winners",
            "--max-invalid-rate", "0",
            "--output", str(canonical),
            "--report", str(conversion_report),
        ]
    )
    run(
        [
            sys.executable,
            str(ROOT / "scripts/build_replay_deck_map.py"),
            "--raw-dir", str(args.replay_dir),
            "--output", str(sidecar),
            "--metadata-output", str(sidecar_report),
        ]
    )
    run(
        [
            sys.executable,
            str(Path(__file__).with_name("build_engine_catalog.py")),
            "--cards", str(args.cards),
            "--attacks", str(args.attacks),
            "--output", str(engine_catalog),
        ]
    )
    run(
        [
            sys.executable,
            str(Path(__file__).with_name("select_initial_decks.py")),
            "--analysis-dir", str(args.analysis_dir),
            "--canonical-decisions", str(canonical),
            "--deck-sidecar", str(sidecar),
            "--target-deck", str(args.target_deck),
            "--output-dir", str(selection_dir),
            "--desired", str(args.desired_decks),
            "--minimum", str(args.minimum_decks),
            "--min-actor-episodes", str(args.min_actor_episodes),
            "--min-policy-decisions", str(args.min_policy_decisions),
        ]
    )
    selected_path = selection_dir / "selected_decks.json"
    run(
        [
            sys.executable,
            str(Path(__file__).with_name("build_from_pocketmon_canonical.py")),
            "--canonical-decisions", str(canonical),
            "--deck-sidecar", str(sidecar),
            "--selected-decks", str(selected_path),
            "--engine-catalog", str(engine_catalog),
            "--output-root", str(dataset_root),
            "--pretrain-max-decisions", str(args.pretrain_max_decisions),
        ]
    )

    manifest_path = dataset_root / "dataset_manifest.json"
    manifest = load(manifest_path)
    sources = [manifest["pretrain"], *manifest["current_sources"]]
    token_script = REFERENCE / "data_pipeline/build_token_cache.py"
    sequence_script = REFERENCE / "data_pipeline/build_sequence_cache.py"
    for source in sources:
        dataset_dir = Path(source["dataset_dir"])
        token_dir = dataset_dir / "token_cache"
        sequence_dir = dataset_dir / "sequence_cache"
        run(
            [
                sys.executable, str(token_script),
                "--decisions", source["decisions"],
                "--features", source["features"],
                "--catalog", str(engine_catalog),
                "--output-dir", str(token_dir),
            ]
        )
        run(
            [
                sys.executable, str(sequence_script),
                "--features", source["features"],
                "--output-dir", str(sequence_dir),
                "--history-length", "8",
            ]
        )
        source["token_cache"] = str(token_dir.resolve())
        source["sequence_cache"] = str(sequence_dir.resolve())

    class_map = dataset_root / "opponent_class_map.json"
    class_command = [
        sys.executable,
        str(REFERENCE / "data_pipeline/build_opponent_deck_class_map.py"),
    ]
    catalog = manifest["catalog"]["path"]
    for source in manifest["current_sources"]:
        class_command.extend(
            [
                "--source",
                source["name"],
                source["decisions"],
                catalog,
                str(source["calibration_episode_count"]),
            ]
        )
    class_command.extend(["--min-train-actors", "3", "--output", str(class_map)])
    run(class_command)

    identity_script = REFERENCE / "data_pipeline/build_deck_identity_cache.py"
    for source in sources:
        dataset_dir = Path(source["dataset_dir"])
        identity_dir = dataset_dir / "identity_cache"
        run(
            [
                sys.executable, str(identity_script),
                "--decisions", source["decisions"],
                "--features", source["features"],
                "--token-cache", source["token_cache"],
                "--catalog", catalog,
                "--class-map", str(class_map),
                "--deck-map", manifest["deck_map"]["path"],
                "--output-dir", str(identity_dir),
            ]
        )
        source["identity_cache"] = str(identity_dir.resolve())

    manifest["class_map"] = {"path": str(class_map.resolve()), "sha256": sha256_file(class_map)}
    manifest["selection"] = {
        "path": str(selected_path.resolve()),
        "sha256": sha256_file(selected_path),
    }
    manifest["conversion_report"] = {
        "path": str(conversion_report.resolve()),
        "sha256": sha256_file(conversion_report),
    }
    manifest["sidecar_report"] = {
        "path": str(sidecar_report.resolve()),
        "sha256": sha256_file(sidecar_report),
    }
    write_json(manifest_path, manifest)
    write_json(
        output / "ready_receipt.json",
        {
            "schema_version": 1,
            "status": "ready_for_training",
            "dataset_manifest": str(manifest_path.resolve()),
            "dataset_manifest_sha256": sha256_file(manifest_path),
            "selected_decks": len(manifest["current_sources"]),
            "canonical_decisions_sha256": sha256_file(canonical),
            "deck_sidecar_sha256": sha256_file(sidecar),
            "engine_catalog_sha256": sha256_file(engine_catalog),
        },
    )
    print(output / "ready_receipt.json")


if __name__ == "__main__":
    main()
