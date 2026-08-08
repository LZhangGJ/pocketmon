#!/usr/bin/env python3
"""Build a weighted exact-deck BC dataset directly from the frozen ZIP catalog."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from features import OPTION_DIM, STATE_DIM, encode_option, encode_state, load_catalog


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--engine-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deck-sha256", required=True)
    parser.add_argument("--module-version", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--require-positive-policy-weight", action="store_true")
    args = parser.parse_args()
    module_versions = {
        value.strip() for value in args.module_version.split(",") if value.strip()
    }
    if not module_versions:
        raise ValueError("module-version leaves an empty allowed set")
    if not 0.0 <= args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be in [0, 1)")

    rows = []
    for row in load_rows(args.catalog):
        if (
            row["is_clean"] != "1"
            or row["module_version"] not in module_versions
            or args.deck_sha256 not in (row["deck0_sha256"], row["deck1_sha256"])
        ):
            continue
        target_players = [
            player
            for player in (0, 1)
            if row[f"deck{player}_sha256"] == args.deck_sha256
            and (
                not args.require_positive_policy_weight
                or float(row[f"policy_weight{player}"]) > 0.0
            )
        ]
        if target_players:
            rows.append(row)
    rows.sort(key=lambda row: (row["create_time"], int(row["episode_id"])))
    if not rows:
        raise ValueError("catalog filter selected no episodes")
    validation_count = (
        0
        if args.validation_fraction == 0.0
        else max(1, math.ceil(len(rows) * args.validation_fraction))
    )
    if validation_count >= len(rows):
        raise ValueError("validation split leaves no training episodes")
    validation_ids = (
        set()
        if validation_count == 0
        else {row["episode_id"] for row in rows[-validation_count:]}
    )
    cards, attacks = load_catalog(args.engine_catalog)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state_features: list[np.ndarray] = []
    option_features: list[np.ndarray] = []
    option_labels: list[int] = []
    option_offsets = [0]
    episode_ids: list[int] = []
    source_steps: list[int] = []
    player_indices: list[int] = []
    min_counts: list[int] = []
    max_counts: list[int] = []
    chosen_counts: list[int] = []
    select_types: list[int] = []
    select_contexts: list[int] = []
    turns: list[int] = []
    validation: list[int] = []
    deck_hashes: list[str] = []
    team_names: list[str] = []
    is_winners: list[int] = []
    policy_weights: list[float] = []
    decision_rows: list[dict[str, Any]] = []
    replay_receipts: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []

    raw_path = args.output_dir / "decisions.jsonl.gz"
    archive_handles: dict[str, zipfile.ZipFile] = {}
    try:
        with gzip.open(raw_path, "wt", encoding="utf-8", newline="\n", compresslevel=6) as raw_handle:
            decision_id = 0
            for row_index, row in enumerate(rows, start=1):
                archive_path = row["source_archive"]
                archive = archive_handles.get(archive_path)
                if archive is None:
                    archive = zipfile.ZipFile(archive_path, "r")
                    archive_handles[archive_path] = archive
                member = f'{row["episode_id"]}.json'
                raw = archive.read(member)
                digest = hashlib.sha256(raw).hexdigest().upper()
                if len(raw) != int(row["json_bytes"]) or digest != row["json_sha256"]:
                    raise RuntimeError(f"source receipt mismatch for episode {row['episode_id']}")
                payload = json.loads(raw)
                if payload.get("module_version") not in module_versions:
                    raise RuntimeError(f"module mismatch for episode {row['episode_id']}")
                replay_receipts.append(
                    {
                        "episodeId": int(row["episode_id"]),
                        "bytes": len(raw),
                        "sha256": digest,
                        "sourceArchive": archive_path,
                        "sourceArchiveSha256": row["source_archive_sha256"],
                    }
                )

                target_players = [
                    player
                    for player in (0, 1)
                    if row[f"deck{player}_sha256"] == args.deck_sha256
                    and (
                        not args.require_positive_policy_weight
                        or float(row[f"policy_weight{player}"]) > 0.0
                    )
                ]
                split_name = "validation" if row["episode_id"] in validation_ids else "train"
                steps = payload.get("steps", [])
                for action_step in range(1, len(steps)):
                    for player_index in target_players:
                        previous = steps[action_step - 1][player_index]
                        current = steps[action_step][player_index]
                        observation = previous.get("observation") or {}
                        select = observation.get("select")
                        if previous.get("status") != "ACTIVE" or select is None:
                            continue
                        action = current.get("action")
                        options = select.get("option") or []
                        min_count = int(select.get("minCount", 0))
                        max_count = int(select.get("maxCount", 0))
                        valid = (
                            isinstance(action, list)
                            and all(isinstance(index, int) and not isinstance(index, bool) for index in action)
                            and len(action) == len(set(action))
                            and min_count <= len(action) <= max_count
                            and all(0 <= index < len(options) for index in action)
                        )
                        if not valid:
                            bad.append(
                                {
                                    "episodeId": row["episode_id"],
                                    "actionStep": action_step,
                                    "playerIndex": player_index,
                                    "action": action,
                                    "minCount": min_count,
                                    "maxCount": max_count,
                                    "optionCount": len(options),
                                }
                            )
                            continue

                        chosen = set(action)
                        state_features.append(encode_state(observation))
                        for option_index, option in enumerate(options):
                            option_features.append(
                                encode_option(observation, option, option_index, cards, attacks)
                            )
                            option_labels.append(int(option_index in chosen))
                        option_offsets.append(len(option_features))
                        episode_ids.append(int(row["episode_id"]))
                        source_steps.append(action_step - 1)
                        player_indices.append(player_index)
                        min_counts.append(min_count)
                        max_counts.append(max_count)
                        chosen_counts.append(len(action))
                        select_types.append(int(select.get("type", -1)))
                        select_contexts.append(int(select.get("context", -1)))
                        turns.append(int((observation.get("current") or {}).get("turn", 0)))
                        validation.append(int(split_name == "validation"))
                        deck_hashes.append(args.deck_sha256)
                        team_names.append(row[f"team{player_index}"])
                        winner = int(row["winner_index"]) == player_index
                        is_winners.append(int(winner))
                        policy_weight = float(row[f"policy_weight{player_index}"])
                        policy_weights.append(policy_weight)

                        raw_record = {
                            "schemaVersion": 2,
                            "decisionId": decision_id,
                            "episodeId": int(row["episode_id"]),
                            "sourceStep": action_step - 1,
                            "actionStep": action_step,
                            "playerIndex": player_index,
                            "teamName": row[f"team{player_index}"],
                            "deckSha256": args.deck_sha256,
                            "split": split_name,
                            "isWinner": winner,
                            "policyWeight": policy_weight,
                            "minScore": float(row["min_score"]),
                            "avgScore": float(row["avg_score"]),
                            "moduleVersion": payload.get("module_version", ""),
                            "observation": observation,
                            "legalOptions": options,
                            "expertSelection": action,
                        }
                        raw_handle.write(json.dumps(raw_record, ensure_ascii=False, separators=(",", ":")))
                        raw_handle.write("\n")
                        decision_rows.append(
                            {
                                "decision_id": decision_id,
                                "episode_id": row["episode_id"],
                                "source_step": action_step - 1,
                                "action_step": action_step,
                                "player_index": player_index,
                                "team_name": row[f"team{player_index}"],
                                "split": split_name,
                                "is_winner": int(winner),
                                "policy_weight": policy_weight,
                                "turn": turns[-1],
                                "select_type": select_types[-1],
                                "select_context": select_contexts[-1],
                                "option_count": len(options),
                                "min_count": min_count,
                                "max_count": max_count,
                                "chosen_count": len(action),
                                "option_offset_begin": option_offsets[-2],
                                "option_offset_end": option_offsets[-1],
                            }
                        )
                        decision_id += 1
                if row_index == 1 or row_index % 25 == 0 or row_index == len(rows):
                    print(f"episode {row_index}/{len(rows)} {row['episode_id']}", flush=True)
    finally:
        for archive in archive_handles.values():
            archive.close()

    if bad:
        raise RuntimeError(f"found {len(bad)} invalid aligned actions; first={bad[0]}")
    if not decision_rows:
        raise RuntimeError("dataset contains no decisions")

    np.savez_compressed(
        args.output_dir / "features.npz",
        state_features=np.asarray(state_features, dtype=np.float32).reshape(-1, STATE_DIM),
        option_features=np.asarray(option_features, dtype=np.float32).reshape(-1, OPTION_DIM),
        option_labels=np.asarray(option_labels, dtype=np.uint8),
        option_offsets=np.asarray(option_offsets, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        source_steps=np.asarray(source_steps, dtype=np.int32),
        player_indices=np.asarray(player_indices, dtype=np.int8),
        min_counts=np.asarray(min_counts, dtype=np.int8),
        max_counts=np.asarray(max_counts, dtype=np.int8),
        chosen_counts=np.asarray(chosen_counts, dtype=np.int8),
        select_types=np.asarray(select_types, dtype=np.int8),
        select_contexts=np.asarray(select_contexts, dtype=np.int8),
        turns=np.asarray(turns, dtype=np.int16),
        validation=np.asarray(validation, dtype=np.uint8),
        deck_hashes=np.asarray(deck_hashes, dtype="U64"),
        team_names=np.asarray(team_names, dtype="U128"),
        is_winners=np.asarray(is_winners, dtype=np.uint8),
        policy_weights=np.asarray(policy_weights, dtype=np.float32),
    )

    with (args.output_dir / "decisions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decision_rows[0]))
        writer.writeheader()
        writer.writerows(decision_rows)
    (args.output_dir / "source_replays.json").write_text(
        json.dumps(replay_receipts, indent=2), encoding="utf-8"
    )

    artifacts = {}
    for path in (raw_path, args.output_dir / "features.npz", args.output_dir / "decisions.csv"):
        artifacts[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        }
    summary = {
        "schemaVersion": 2,
        "deckSha256": args.deck_sha256,
        "moduleVersions": sorted(module_versions),
        "sourceUniqueEpisodes": len(rows),
        "sourceActorEpisodes": sum(
            int(
                row["deck0_sha256"] == args.deck_sha256
                and (
                    not args.require_positive_policy_weight
                    or float(row["policy_weight0"]) > 0.0
                )
            )
            + int(
                row["deck1_sha256"] == args.deck_sha256
                and (
                    not args.require_positive_policy_weight
                    or float(row["policy_weight1"]) > 0.0
                )
            )
            for row in rows
        ),
        "trainEpisodes": len(rows) - validation_count,
        "validationEpisodes": validation_count,
        "decisions": len(decision_rows),
        "trainDecisions": sum(1 - value for value in validation),
        "validationDecisions": sum(validation),
        "winnerDecisions": sum(is_winners),
        "strongLoserDecisions": len(is_winners) - sum(is_winners),
        "legalOptions": len(option_features),
        "expertSelections": int(sum(option_labels)),
        "invalidAlignedActions": 0,
        "stateFeatureDimension": STATE_DIM,
        "optionFeatureDimension": OPTION_DIM,
        "teamDecisionCounts": Counter(team_names),
        "selectContextCounts": Counter(select_contexts),
        "policyWeightCounts": Counter(str(value) for value in policy_weights),
        "artifacts": artifacts,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
