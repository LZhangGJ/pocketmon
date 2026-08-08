from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    Experiment7Error,
    is_forced_decision,
    read_csv,
    sha256_file,
    stable_episode_key,
    utc_now,
    validate_action,
    write_csv,
    write_json,
)


def load_safe_feature_api(reference_root: Path):
    vendor_path = reference_root / "data_pipeline" / "features.py"
    if not vendor_path.is_file():
        raise FileNotFoundError(vendor_path)
    spec = importlib.util.spec_from_file_location("features_vendor", vendor_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {vendor_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["features_vendor"] = module
    spec.loader.exec_module(module)
    integration_root = Path(__file__).resolve().parent
    if str(integration_root) not in sys.path:
        sys.path.insert(0, str(integration_root))
    import safe_features  # type: ignore

    return safe_features


def parse_modules(value: str) -> set[str] | None:
    if value.strip() in {"", "*", "all"}:
        return None
    result = {token.strip() for token in value.split(",") if token.strip()}
    if not result:
        raise ValueError("module version filter is empty")
    return result


def selected_rows(
    catalog: Path,
    mode: str,
    deck_sha256: str | None,
    modules: set[str] | None,
    positive_policy_only: bool,
    max_episodes: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(catalog):
        if row.get("is_clean") != "1":
            continue
        if modules is not None and row.get("module_version") not in modules:
            continue
        if mode == "exact" and deck_sha256 not in (
            row.get("deck0_sha256", "").lower(),
            row.get("deck1_sha256", "").lower(),
        ):
            continue
        target_players = []
        for player in (0, 1):
            if mode == "exact" and row.get(f"deck{player}_sha256", "").lower() != deck_sha256:
                continue
            if positive_policy_only and float(row.get(f"policy_weight{player}", 0.0) or 0.0) <= 0:
                continue
            target_players.append(player)
        if target_players:
            copied = dict(row)
            copied["_target_players"] = ",".join(str(player) for player in target_players)
            rows.append(copied)
    rows.sort(key=stable_episode_key)
    if max_episodes > 0:
        rows = rows[:max_episodes]
    if not rows:
        raise Experiment7Error("dataset filter selected no episodes")
    return rows


def build_dataset(
    *,
    reference_root: Path,
    catalog: Path,
    engine_catalog: Path,
    output_dir: Path,
    mode: str,
    deck_sha256: str | None,
    module_versions: str,
    validation_fraction: float,
    positive_policy_only: bool,
    skip_forced: bool,
    max_episodes: int,
) -> dict[str, Any]:
    if mode == "exact" and not deck_sha256:
        raise ValueError("exact mode requires deck_sha256")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    modules = parse_modules(module_versions)
    rows = selected_rows(
        catalog,
        mode,
        deck_sha256.lower() if deck_sha256 else None,
        modules,
        positive_policy_only,
        max_episodes,
    )
    validation_count = (
        0 if validation_fraction == 0.0 else max(1, math.ceil(len(rows) * validation_fraction))
    )
    if validation_count >= len(rows):
        raise Experiment7Error("validation split leaves no training episodes")
    validation_ids = (
        {int(row["episode_id"]) for row in rows[-validation_count:]}
        if validation_count
        else set()
    )

    safe_features = load_safe_feature_api(reference_root)
    cards, attacks = safe_features.load_catalog(engine_catalog)
    state_dim = int(safe_features.STATE_DIM)
    option_dim = int(safe_features.OPTION_DIM)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    invalid: list[dict[str, Any]] = []
    skipped_forced = 0
    raw_path = output_dir / "decisions.jsonl.gz"

    with gzip.open(raw_path, "wt", encoding="utf-8", newline="\n", compresslevel=6) as raw_handle:
        decision_id = 0
        for row_index, row in enumerate(rows, start=1):
            replay_path = Path(row["raw_path"])
            raw = replay_path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if len(raw) != int(row["json_bytes"]) or digest != row["json_sha256"]:
                raise Experiment7Error(f"source receipt mismatch for episode {row['episode_id']}")
            payload = json.loads(raw)
            if modules is not None and str(payload.get("module_version") or "") not in modules:
                raise Experiment7Error(f"module mismatch for episode {row['episode_id']}")
            replay_receipts.append(
                {
                    "episodeId": int(row["episode_id"]),
                    "bytes": len(raw),
                    "sha256": digest,
                    "rawPath": str(replay_path.resolve()),
                }
            )
            target_players = [int(value) for value in row["_target_players"].split(",")]
            split_name = "validation" if int(row["episode_id"]) in validation_ids else "train"
            steps = payload.get("steps") or []
            for action_step in range(1, len(steps)):
                previous_pair = steps[action_step - 1]
                current_pair = steps[action_step]
                if not isinstance(previous_pair, list) or not isinstance(current_pair, list):
                    continue
                for player_index in target_players:
                    if player_index >= len(previous_pair) or player_index >= len(current_pair):
                        continue
                    previous = previous_pair[player_index]
                    current = current_pair[player_index]
                    if not isinstance(previous, dict) or not isinstance(current, dict):
                        continue
                    observation = previous.get("observation") or {}
                    select = observation.get("select") if isinstance(observation, dict) else None
                    if previous.get("status") != "ACTIVE" or not isinstance(select, dict):
                        continue
                    options = select.get("option") or []
                    minimum = int(select.get("minCount", 0) or 0)
                    maximum = int(select.get("maxCount", 0) or 0)
                    action = current.get("action")
                    valid, reason = validate_action(action, options, minimum, maximum)
                    if not valid:
                        invalid.append(
                            {
                                "episodeId": int(row["episode_id"]),
                                "actionStep": action_step,
                                "playerIndex": player_index,
                                "reason": reason,
                                "action": action,
                                "minCount": minimum,
                                "maxCount": maximum,
                                "optionCount": len(options),
                            }
                        )
                        continue
                    if skip_forced and is_forced_decision(options, minimum, maximum):
                        skipped_forced += 1
                        continue

                    sanitized = safe_features.sanitize_observation(observation)
                    sanitized_select = sanitized.get("select") or {}
                    sanitized_options = sanitized_select.get("option") or []
                    if len(sanitized_options) != len(options):
                        raise Experiment7Error("sanitization changed legal option count")
                    chosen = set(int(index) for index in action)
                    state_features.append(safe_features.encode_state(sanitized))
                    for option_index, option in enumerate(sanitized_options):
                        option_features.append(
                            safe_features.encode_option(
                                sanitized, option, option_index, cards, attacks
                            )
                        )
                        option_labels.append(int(option_index in chosen))
                    option_offsets.append(len(option_features))
                    episode = int(row["episode_id"])
                    episode_ids.append(episode)
                    source_steps.append(action_step - 1)
                    player_indices.append(player_index)
                    min_counts.append(minimum)
                    max_counts.append(maximum)
                    chosen_counts.append(len(action))
                    select_types.append(int(select.get("type", -1)))
                    select_contexts.append(int(select.get("context", -1)))
                    turns.append(int((sanitized.get("current") or {}).get("turn", 0) or 0))
                    validation.append(int(split_name == "validation"))
                    own_hash = row[f"deck{player_index}_sha256"].lower()
                    deck_hashes.append(own_hash)
                    team_names.append(row.get(f"team{player_index}", ""))
                    winner = int(row["winner_index"]) == player_index
                    is_winners.append(int(winner))
                    policy_weight = float(row.get(f"policy_weight{player_index}", 0.0) or 0.0)
                    policy_weights.append(policy_weight)
                    raw_record = {
                        "schemaVersion": 3,
                        "decisionId": decision_id,
                        "episodeId": episode,
                        "sourceStep": action_step - 1,
                        "actionStep": action_step,
                        "playerIndex": player_index,
                        "teamName": row.get(f"team{player_index}", ""),
                        "deckSha256": own_hash,
                        "split": split_name,
                        "isWinner": winner,
                        "policyWeight": policy_weight,
                        "minScore": float(row.get("min_score", 0.0) or 0.0),
                        "avgScore": float(row.get("avg_score", 0.0) or 0.0),
                        "moduleVersion": str(payload.get("module_version") or ""),
                        "observation": sanitized,
                        "legalOptions": sanitized_options,
                        "expertSelection": [int(index) for index in action],
                    }
                    raw_handle.write(json.dumps(raw_record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    decision_rows.append(
                        {
                            "decision_id": decision_id,
                            "episode_id": episode,
                            "source_step": action_step - 1,
                            "action_step": action_step,
                            "player_index": player_index,
                            "team_name": row.get(f"team{player_index}", ""),
                            "split": split_name,
                            "is_winner": int(winner),
                            "policy_weight": policy_weight,
                            "turn": turns[-1],
                            "select_type": select_types[-1],
                            "select_context": select_contexts[-1],
                            "option_count": len(sanitized_options),
                            "min_count": minimum,
                            "max_count": maximum,
                            "chosen_count": len(action),
                            "option_offset_begin": option_offsets[-2],
                            "option_offset_end": option_offsets[-1],
                        }
                    )
                    decision_id += 1
            if row_index == 1 or row_index % 50 == 0 or row_index == len(rows):
                print(json.dumps({"episodes": row_index, "totalEpisodes": len(rows), "decisions": decision_id}), flush=True)

    if invalid:
        write_json(output_dir / "invalid_actions.json", invalid)
        raise Experiment7Error(f"found {len(invalid)} invalid aligned actions; first={invalid[0]}")
    if not decision_rows:
        raise Experiment7Error("dataset contains no non-forced decisions")

    np.savez_compressed(
        output_dir / "features.npz",
        state_features=np.asarray(state_features, dtype=np.float32).reshape(-1, state_dim),
        option_features=np.asarray(option_features, dtype=np.float32).reshape(-1, option_dim),
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
    decisions_path = output_dir / "decisions.csv"
    write_csv(decisions_path, decision_rows)
    write_json(output_dir / "source_replays.json", replay_receipts)
    features_path = output_dir / "features.npz"
    summary = {
        "schemaVersion": 3,
        "createdAt": utc_now(),
        "mode": mode,
        "deckSha256": deck_sha256,
        "moduleVersions": sorted(modules) if modules is not None else ["*"],
        "sourceEpisodes": len(rows),
        "trainEpisodes": len(rows) - validation_count,
        "validationEpisodes": validation_count,
        "decisions": len(decision_rows),
        "trainDecisions": int(sum(1 - value for value in validation)),
        "validationDecisions": int(sum(validation)),
        "winnerDecisions": int(sum(is_winners)),
        "legalOptions": len(option_features),
        "expertSelections": int(sum(option_labels)),
        "skippedForcedDecisions": skipped_forced,
        "invalidAlignedActions": 0,
        "stateFeatureDimension": state_dim,
        "optionFeatureDimension": option_dim,
        "teamDecisionCounts": dict(Counter(team_names)),
        "selectContextCounts": {str(key): int(value) for key, value in Counter(select_contexts).items()},
        "artifacts": {
            "decisions.jsonl.gz": {"bytes": raw_path.stat().st_size, "sha256": sha256_file(raw_path)},
            "features.npz": {"bytes": features_path.stat().st_size, "sha256": sha256_file(features_path)},
            "decisions.csv": {"bytes": decisions_path.stat().st_size, "sha256": sha256_file(decisions_path)},
        },
        "privacyBoundary": "observations are sanitized before feature/token cache generation; opponent hand and all prize identities are removed while public counts remain",
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Experiment 7 features directly from pocketmon raw replays")
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--engine-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("broad", "exact"), required=True)
    parser.add_argument("--deck-sha256")
    parser.add_argument("--module-version", default="*")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--all-players", action="store_true")
    parser.add_argument("--include-forced", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()
    build_dataset(
        reference_root=args.reference_root.resolve(),
        catalog=args.catalog.resolve(),
        engine_catalog=args.engine_catalog.resolve(),
        output_dir=args.output_dir.resolve(),
        mode=args.mode,
        deck_sha256=args.deck_sha256,
        module_versions=args.module_version,
        validation_fraction=args.validation_fraction,
        positive_policy_only=not args.all_players,
        skip_forced=not args.include_forced,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
