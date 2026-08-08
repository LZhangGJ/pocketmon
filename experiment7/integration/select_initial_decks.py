from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import (
    canonical_deck_sha256,
    load_csv,
    open_jsonl,
    parse_float,
    parse_int,
    read_deck,
    safe_slug,
    sha256_file,
    signature_id,
    write_deck,
    write_json,
)


def _load_representatives(path: Path) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in load_csv(path):
        archetype_id = row["archetype_id"]
        record = grouped.setdefault(
            archetype_id,
            {
                "archetype_id": archetype_id,
                "archetype_rank": parse_int(row.get("archetype_rank"), 10**9),
                "archetype_label": row["archetype_label"],
                "representative_exact_deck_id": row["representative_exact_deck_id"],
                "cards": Counter(),
            },
        )
        if record["representative_exact_deck_id"] != row["representative_exact_deck_id"]:
            raise ValueError(f"{archetype_id}: multiple representative deck IDs")
        card_id = parse_int(row.get("card_id"), 0)
        count = parse_int(row.get("count"), 0)
        if card_id <= 0 or count <= 0:
            raise ValueError(f"invalid representative deck row: {row}")
        record["cards"][card_id] += count
    for record in grouped.values():
        cards = [card for card, count in sorted(record["cards"].items()) for _ in range(count)]
        if len(cards) != 60:
            raise ValueError(
                f"{record['archetype_id']}: representative deck has {len(cards)} cards, expected 60"
            )
        record["cards"] = cards
        record["deck_sha256"] = canonical_deck_sha256(cards)
        record["computed_exact_deck_id"] = signature_id(cards)
    return grouped


def _load_support(
    canonical: Path,
    deck_sidecar: Path,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, int], str]]:
    actor_decks: dict[tuple[str, int], str] = {}
    support: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "actor_episodes": set(),
            "policy_actor_episodes": set(),
            "valid_decisions": 0,
            "nonforced_policy_decisions": 0,
            "first_create_time": None,
            "last_create_time": None,
        }
    )
    for row in open_jsonl(deck_sidecar):
        episode = str(row["episode_id"])
        player = int(row["player"])
        deck_hash = canonical_deck_sha256(row["deck"])
        key = (episode, player)
        previous = actor_decks.setdefault(key, deck_hash)
        if previous != deck_hash:
            raise RuntimeError(f"conflicting deck sidecar entry for {key}")
        support[deck_hash]["actor_episodes"].add(key)

    for row in open_jsonl(canonical):
        key = (str(row["episode_id"]), int(row["player"]))
        deck_hash = actor_decks.get(key)
        if deck_hash is None:
            raise RuntimeError(f"canonical decision has no deck sidecar: {key}")
        bucket = support[deck_hash]
        bucket["valid_decisions"] += 1
        manifest = row.get("manifest") or {}
        create_time = str(manifest.get("create_time") or manifest.get("CreateTime") or "")
        if create_time:
            bucket["first_create_time"] = min(
                value for value in (bucket["first_create_time"], create_time) if value
            )
            bucket["last_create_time"] = max(
                value for value in (bucket["last_create_time"], create_time) if value
            )
        observation = row.get("observation") or {}
        select = observation.get("select") or {}
        options = select.get("option") or []
        minimum = parse_int(select.get("minCount"), 0)
        maximum = parse_int(select.get("maxCount"), minimum)
        forced = minimum == maximum and (minimum == 0 or minimum == len(options))
        if float(row.get("policy_weight") or 0.0) > 0.0:
            bucket["policy_actor_episodes"].add(key)
            if not forced:
                bucket["nonforced_policy_decisions"] += 1

    normalized: dict[str, dict[str, Any]] = {}
    for deck_hash, bucket in support.items():
        normalized[deck_hash] = {
            **{key: value for key, value in bucket.items() if not isinstance(value, set)},
            "actor_episodes": len(bucket["actor_episodes"]),
            "policy_actor_episodes": len(bucket["policy_actor_episodes"]),
        }
    return normalized, actor_decks


def _score(row: dict[str, Any]) -> tuple[float, float, int, int]:
    # The ladder report's score level is the primary signal.  Conservative
    # non-mirror performance and sample size break ties.
    return (
        parse_float(row.get("avg_game_min_score"), 0.0),
        parse_float(row.get("nonmirror_ci_low"), 0.0),
        parse_int(row.get("nonmirror_appearances"), 0),
        -parse_int(row.get("rank"), 10**9),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select supported exact 60-card challengers from the frozen top-ladder report"
    )
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--canonical-decisions", type=Path, required=True)
    parser.add_argument("--deck-sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-deck", type=Path)
    parser.add_argument("--desired", type=int, default=6)
    parser.add_argument("--minimum", type=int, default=4)
    parser.add_argument("--min-actor-episodes", type=int, default=40)
    parser.add_argument("--min-policy-decisions", type=int, default=800)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if not 1 <= args.minimum <= args.desired:
        raise ValueError("require 1 <= minimum <= desired")
    if not 0.0 < args.holdout_fraction < 0.5:
        raise ValueError("holdout-fraction must be in (0, 0.5)")
    if not 0.0 < args.calibration_fraction < 0.5:
        raise ValueError("calibration-fraction must be in (0, 0.5)")

    archetypes = {
        row["archetype_id"]: row
        for row in load_csv(args.analysis_dir / "archetype_summary.csv")
    }
    representatives = _load_representatives(
        args.analysis_dir / "representative_decklists.csv"
    )
    support, _ = _load_support(args.canonical_decisions, args.deck_sidecar)
    target_hash = (
        canonical_deck_sha256(read_deck(args.target_deck))
        if args.target_deck and args.target_deck.is_file()
        else None
    )

    candidates: list[dict[str, Any]] = []
    for archetype_id, representative in representatives.items():
        summary = archetypes.get(archetype_id)
        if summary is None:
            raise ValueError(f"{archetype_id}: missing archetype summary")
        deck_hash = representative["deck_sha256"]
        deck_support = support.get(
            deck_hash,
            {
                "actor_episodes": 0,
                "policy_actor_episodes": 0,
                "valid_decisions": 0,
                "nonforced_policy_decisions": 0,
                "first_create_time": None,
                "last_create_time": None,
            },
        )
        actor_episodes = int(deck_support["policy_actor_episodes"])
        training_pool = max(0, actor_episodes - max(1, math.ceil(actor_episodes * args.holdout_fraction)))
        calibration_episodes = (
            max(1, math.ceil(training_pool * args.calibration_fraction))
            if training_pool >= 2
            else 0
        )
        qualified = (
            deck_hash != target_hash
            and actor_episodes >= args.min_actor_episodes
            and int(deck_support["nonforced_policy_decisions"]) >= args.min_policy_decisions
            and training_pool > calibration_episodes > 0
        )
        reason = "qualified"
        if deck_hash == target_hash:
            reason = "same_exact_deck_as_primary_target"
        elif actor_episodes < args.min_actor_episodes:
            reason = "insufficient_exact_deck_actor_episodes"
        elif int(deck_support["nonforced_policy_decisions"]) < args.min_policy_decisions:
            reason = "insufficient_nonforced_policy_decisions"
        elif not (training_pool > calibration_episodes > 0):
            reason = "chronological_split_not_viable"
        candidates.append(
            {
                **summary,
                **{key: value for key, value in representative.items() if key != "cards"},
                **deck_support,
                "calibration_episode_count": calibration_episodes,
                "qualified": qualified,
                "qualification_reason": reason,
                "_cards": representative["cards"],
            }
        )

    eligible = sorted(
        (row for row in candidates if row["qualified"]),
        key=_score,
        reverse=True,
    )
    selected = eligible[: args.desired]
    if len(selected) < args.minimum:
        raise RuntimeError(
            f"only {len(selected)} ladder decks passed support gates; minimum is {args.minimum}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    deck_dir = args.output_dir / "decks"
    manifest_rows = []
    for index, row in enumerate(selected, 1):
        slug = f"{index:02d}-{safe_slug(str(row['archetype_label']))}"
        deck_path = deck_dir / f"{slug}.csv"
        write_deck(deck_path, row["_cards"])
        manifest_rows.append(
            {
                "index": index,
                "name": slug,
                "archetype_id": row["archetype_id"],
                "archetype_label": row["archetype_label"],
                "ladder_rank": parse_int(row.get("rank"), 0),
                "avg_game_min_score": parse_float(row.get("avg_game_min_score"), 0.0),
                "nonmirror_win_rate": parse_float(row.get("nonmirror_win_rate"), 0.0),
                "nonmirror_ci_low": parse_float(row.get("nonmirror_ci_low"), 0.0),
                "appearances": parse_int(row.get("appearances"), 0),
                "representative_exact_deck_id": row["representative_exact_deck_id"],
                "deck_sha256": row["deck_sha256"],
                "deck_path": str(deck_path.resolve()),
                "deck_file_sha256": sha256_file(deck_path),
                "actor_episodes": int(row["actor_episodes"]),
                "policy_actor_episodes": int(row["policy_actor_episodes"]),
                "valid_decisions": int(row["valid_decisions"]),
                "nonforced_policy_decisions": int(row["nonforced_policy_decisions"]),
                "calibration_episode_count": int(row["calibration_episode_count"]),
            }
        )

    candidate_csv = args.output_dir / "candidate_table.csv"
    public_fields = [
        "rank",
        "archetype_id",
        "archetype_label",
        "avg_game_min_score",
        "nonmirror_win_rate",
        "nonmirror_ci_low",
        "appearances",
        "representative_exact_deck_id",
        "deck_sha256",
        "actor_episodes",
        "policy_actor_episodes",
        "valid_decisions",
        "nonforced_policy_decisions",
        "calibration_episode_count",
        "qualified",
        "qualification_reason",
    ]
    with candidate_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=public_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(candidates, key=lambda row: parse_int(row.get("rank"), 10**9)))

    payload = {
        "schema_version": 1,
        "selection_rule": (
            "filter by exact-deck replay support and chronological split viability; "
            "rank qualified representatives by avg_game_min_score, conservative nonmirror rate, and support"
        ),
        "analysis_dir": str(args.analysis_dir.resolve()),
        "analysis_inputs": {
            "archetype_summary.csv": sha256_file(args.analysis_dir / "archetype_summary.csv"),
            "representative_decklists.csv": sha256_file(args.analysis_dir / "representative_decklists.csv"),
        },
        "canonical_decisions": {
            "path": str(args.canonical_decisions.resolve()),
            "sha256": sha256_file(args.canonical_decisions),
        },
        "deck_sidecar": {
            "path": str(args.deck_sidecar.resolve()),
            "sha256": sha256_file(args.deck_sidecar),
        },
        "target_deck_sha256": target_hash,
        "thresholds": {
            "desired": args.desired,
            "minimum": args.minimum,
            "min_actor_episodes": args.min_actor_episodes,
            "min_nonforced_policy_decisions": args.min_policy_decisions,
            "holdout_fraction": args.holdout_fraction,
            "calibration_fraction": args.calibration_fraction,
        },
        "selected": manifest_rows,
        "candidate_table": str(candidate_csv.resolve()),
    }
    write_json(args.output_dir / "selected_decks.json", payload)
    print(f"selected {len(manifest_rows)} decks -> {args.output_dir / 'selected_decks.json'}")


if __name__ == "__main__":
    main()
