from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from common import (
    Experiment7Error,
    canonical_deck,
    canonical_deck_sha256,
    find_unique_file,
    read_csv,
    read_deck,
    sha256_file,
    utc_now,
    write_csv,
    write_deck,
    write_json,
)


def _float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default) or default))
    except (TypeError, ValueError):
        return default


def _normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _pokemon_core(rows: Sequence[Mapping[str, str]]) -> frozenset[int]:
    return frozenset(
        int(row["card_id"])
        for row in rows
        if row.get("card_group", "").strip().lower() in {"pokémon", "pokemon"}
        and int(row.get("count", "0") or 0) > 0
    )


def _jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def load_representatives(path: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        grouped[(row["archetype_id"], row["representative_exact_deck_id"])].append(row)
    values: list[dict[str, Any]] = []
    for (archetype_id, exact_id), rows in grouped.items():
        cards: list[int] = []
        for row in rows:
            cards.extend([int(row["card_id"])] * int(row["count"]))
        deck = canonical_deck(cards)
        values.append(
            {
                "archetypeId": archetype_id,
                "archetypeRank": int(rows[0]["archetype_rank"]),
                "archetypeLabel": rows[0]["archetype_label"],
                "representativeExactDeckId": exact_id,
                "cards": deck,
                "deckSha256": canonical_deck_sha256(deck),
                "pokemonCore": _pokemon_core(rows),
            }
        )
    return values


def support_by_deck(catalog_path: Path, module_version: str | None = None) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "actorEpisodes": 0,
            "policyActorEpisodes": 0,
            "nonforcedDecisions": 0,
            "policyDecisions": 0,
            "moduleCounts": Counter(),
            "latestTimestamp": 0.0,
        }
    )
    for row in read_csv(catalog_path):
        if row.get("is_clean") != "1":
            continue
        if module_version and row.get("module_version") != module_version:
            continue
        for player in (0, 1):
            deck_hash = row.get(f"deck{player}_sha256", "").strip().lower()
            if not deck_hash:
                continue
            entry = result[deck_hash]
            entry["actorEpisodes"] += 1
            policy_weight = _float(row, f"policy_weight{player}")
            if policy_weight > 0:
                entry["policyActorEpisodes"] += 1
            entry["nonforcedDecisions"] += _int(row, f"nonforced_decisions{player}")
            entry["policyDecisions"] += _int(row, f"policy_decisions{player}")
            entry["moduleCounts"][row.get("module_version", "")] += 1
            entry["latestTimestamp"] = max(
                float(entry["latestTimestamp"]), _float(row, "create_timestamp")
            )
    return result


def choose_decks(
    ladder_dir: Path,
    catalog_path: Path,
    output_dir: Path,
    target_deck: Path,
    desired: int,
    minimum: int,
    min_actor_episodes: int,
    min_policy_decisions: int,
    near_duplicate_threshold: float,
    exclude_label: str,
    module_version: str | None,
) -> dict[str, Any]:
    summary_path = find_unique_file(ladder_dir, "archetype_summary.csv")
    representative_path = find_unique_file(ladder_dir, "representative_decklists.csv")
    summary = {row["archetype_id"]: row for row in read_csv(summary_path)}
    support = support_by_deck(catalog_path, module_version)
    target_hash = canonical_deck_sha256(read_deck(target_deck))
    candidates = []
    for representative in load_representatives(representative_path):
        row = summary.get(representative["archetypeId"], {})
        deck_support = support.get(representative["deckSha256"], {})
        label = representative["archetypeLabel"]
        target_collision = representative["deckSha256"] == target_hash
        label_excluded = bool(exclude_label and exclude_label.lower() in label.lower())
        actor_episodes = int(deck_support.get("policyActorEpisodes", 0))
        policy_decisions = int(deck_support.get("policyDecisions", 0))
        eligible = (
            not target_collision
            and not label_excluded
            and actor_episodes >= min_actor_episodes
            and policy_decisions >= min_policy_decisions
        )
        candidates.append(
            {
                **representative,
                "cards": list(representative["cards"]),
                "pokemonCore": sorted(representative["pokemonCore"]),
                "appearances": _int(row, "appearances"),
                "usageShare": _float(row, "usage_share"),
                "nonmirrorWinRate": _float(row, "nonmirror_win_rate"),
                "nonmirrorCiLow": _float(row, "nonmirror_ci_low"),
                "avgGameMinScore": _float(row, "avg_game_min_score"),
                "exactDeckVariants": _int(row, "exact_deck_variants"),
                "actorEpisodes": int(deck_support.get("actorEpisodes", 0)),
                "policyActorEpisodes": actor_episodes,
                "nonforcedDecisions": int(deck_support.get("nonforcedDecisions", 0)),
                "policyDecisions": policy_decisions,
                "moduleCounts": dict(deck_support.get("moduleCounts", {})),
                "targetCollision": target_collision,
                "labelExcluded": label_excluded,
                "eligible": eligible,
            }
        )

    ladder_raw = [
        0.45 * candidate["nonmirrorCiLow"]
        + 0.30 * candidate["nonmirrorWinRate"]
        + 0.15 * math.log1p(candidate["appearances"])
        + 0.10 * math.log1p(max(candidate["avgGameMinScore"], 0.0))
        for candidate in candidates
    ]
    support_raw = [
        math.log1p(candidate["policyActorEpisodes"])
        + math.log1p(candidate["policyDecisions"]) / 2.0
        for candidate in candidates
    ]
    ladder_norm = _normalize(ladder_raw)
    support_norm = _normalize(support_raw)
    for index, candidate in enumerate(candidates):
        candidate["ladderScore"] = ladder_norm[index]
        candidate["supportScore"] = support_norm[index]
        candidate["selectionScore"] = 0.60 * ladder_norm[index] + 0.40 * support_norm[index]

    pool = sorted(
        [candidate for candidate in candidates if candidate["eligible"]],
        key=lambda candidate: (
            -float(candidate["selectionScore"]),
            int(candidate["archetypeRank"]),
            candidate["archetypeId"],
        ),
    )
    selected: list[dict[str, Any]] = []
    for candidate in pool:
        core = frozenset(candidate["pokemonCore"])
        maximum_similarity = max(
            (_jaccard(core, frozenset(item["pokemonCore"])) for item in selected),
            default=0.0,
        )
        candidate["maxSelectedCoreJaccardAtDecision"] = maximum_similarity
        if selected and maximum_similarity >= near_duplicate_threshold:
            candidate["selectionStatus"] = "near_duplicate"
            continue
        candidate["selectionStatus"] = "selected"
        selected.append(candidate)
        if len(selected) >= desired:
            break

    if len(selected) < desired:
        selected_hashes = {candidate["deckSha256"] for candidate in selected}
        for candidate in pool:
            if candidate["deckSha256"] in selected_hashes:
                continue
            candidate["selectionStatus"] = "selected_after_diversity_relaxation"
            selected.append(candidate)
            selected_hashes.add(candidate["deckSha256"])
            if len(selected) >= desired:
                break

    if len(selected) < minimum:
        raise Experiment7Error(
            f"only {len(selected)} eligible high-ladder exact decks; minimum is {minimum}. "
            "Inspect deck_candidate_table.csv instead of silently lowering thresholds."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    deck_output = output_dir / "decks"
    selected_rows = []
    for rank, candidate in enumerate(selected, start=1):
        slug = f"{rank:02d}_{candidate['archetypeId'].lower()}_{candidate['deckSha256'][:12]}"
        deck_path = write_deck(deck_output / f"{slug}.csv", candidate["cards"])
        candidate["name"] = slug
        candidate["deckPath"] = str(deck_path.resolve())
        candidate["deckFileSha256"] = sha256_file(deck_path)
        selected_rows.append(
            {
                "selection_rank": rank,
                "name": slug,
                "archetype_id": candidate["archetypeId"],
                "archetype_label": candidate["archetypeLabel"],
                "deck_sha256": candidate["deckSha256"],
                "deck_path": str(deck_path.resolve()),
                "ladder_rank": candidate["archetypeRank"],
                "appearances": candidate["appearances"],
                "nonmirror_win_rate": candidate["nonmirrorWinRate"],
                "nonmirror_ci_low": candidate["nonmirrorCiLow"],
                "policy_actor_episodes": candidate["policyActorEpisodes"],
                "policy_decisions": candidate["policyDecisions"],
                "selection_score": candidate["selectionScore"],
                "selection_status": candidate["selectionStatus"],
            }
        )
    write_csv(output_dir / "selected_decks.csv", selected_rows)

    candidate_fields = [
        "archetypeRank",
        "archetypeId",
        "archetypeLabel",
        "representativeExactDeckId",
        "deckSha256",
        "appearances",
        "usageShare",
        "nonmirrorWinRate",
        "nonmirrorCiLow",
        "avgGameMinScore",
        "actorEpisodes",
        "policyActorEpisodes",
        "nonforcedDecisions",
        "policyDecisions",
        "ladderScore",
        "supportScore",
        "selectionScore",
        "targetCollision",
        "labelExcluded",
        "eligible",
        "selectionStatus",
    ]
    write_csv(
        output_dir / "deck_candidate_table.csv",
        [{field: candidate.get(field, "") for field in candidate_fields} for candidate in candidates],
        candidate_fields,
    )

    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "ladderDirectory": str(ladder_dir.resolve()),
        "ladderSources": {
            "archetypeSummary": {"path": str(summary_path.resolve()), "sha256": sha256_file(summary_path)},
            "representativeDecklists": {"path": str(representative_path.resolve()), "sha256": sha256_file(representative_path)},
        },
        "catalog": {"path": str(catalog_path.resolve()), "sha256": sha256_file(catalog_path)},
        "targetDeck": {"path": str(target_deck.resolve()), "sha256": sha256_file(target_deck), "canonicalDeckSha256": target_hash},
        "selectionRule": {
            "desired": desired,
            "minimum": minimum,
            "minPolicyActorEpisodes": min_actor_episodes,
            "minPolicyDecisions": min_policy_decisions,
            "nearDuplicatePokemonCoreJaccard": near_duplicate_threshold,
            "excludeLabelSubstring": exclude_label,
            "moduleVersion": module_version,
            "usesSealedHoldout": False,
            "usesTargetArenaResults": False,
        },
        "selected": selected,
    }
    write_json(output_dir / "selected_decks.json", payload)
    print(json.dumps({"selected": selected_rows, "count": len(selected_rows)}, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Select high-ladder exact decks with real replay support")
    parser.add_argument("--ladder-dir", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--target-deck", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--desired", type=int, default=6)
    parser.add_argument("--minimum", type=int, default=4)
    parser.add_argument("--min-actor-episodes", type=int, default=10)
    parser.add_argument("--min-policy-decisions", type=int, default=500)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.80)
    parser.add_argument("--exclude-label", default="Mega Lucario")
    parser.add_argument("--module-version")
    args = parser.parse_args()
    if not 1 <= args.minimum <= args.desired:
        raise ValueError("require 1 <= minimum <= desired")
    choose_decks(
        args.ladder_dir.resolve(),
        args.catalog.resolve(),
        args.output_dir.resolve(),
        args.target_deck.resolve(),
        args.desired,
        args.minimum,
        args.min_actor_episodes,
        args.min_policy_decisions,
        args.near_duplicate_threshold,
        args.exclude_label,
        args.module_version,
    )


if __name__ == "__main__":
    main()
