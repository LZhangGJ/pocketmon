from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


STRICT_PREDICATE = "is_clean == 1 and float(min_score) > 1000.0"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalized_card_names(card_names: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({" ".join(name.casefold().split()) for name in card_names}))


def has_family(card_names: Iterable[str], family: str) -> bool:
    needle = " ".join(family.casefold().split())
    return any(needle in name for name in normalized_card_names(card_names))


def matching_archetypes(card_names: Iterable[str], config: dict[str, Any]) -> list[str]:
    names = normalized_card_names(card_names)
    matches: list[tuple[int, str]] = []
    for row in config["archetypes"]:
        required = row.get("requireAll", [])
        excluded = row.get("excludeAny", [])
        if all(has_family(names, value) for value in required) and not any(
            has_family(names, value) for value in excluded
        ):
            matches.append((int(row.get("priority", 0)), str(row["id"])))
    if not matches:
        return []
    # Specific composite archetypes may intentionally share a base family.  The
    # highest priority is authoritative, while equal-priority ties stay visible
    # and are rejected by the caller as ambiguous rather than silently duplicated.
    priority = max(value for value, _ in matches)
    return sorted(name for value, name in matches if value == priority)


def strict_row(row: dict[str, str], threshold: float = 1000.0) -> bool:
    return row.get("is_clean") == "1" and float(row["min_score"]) > threshold


def validation_episode(episode_id: str, config: dict[str, Any]) -> bool:
    split = config["stableSplit"]
    digest = hashlib.sha256(str(episode_id).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return value % int(split["validationModulo"]) == int(split["validationResidue"])


def read_deck_card_names(catalog_dir: Path, engine_catalog: Path) -> dict[str, tuple[str, ...]]:
    cards = load_json(engine_catalog)["cards"]
    card_name = {str(row["cardId"]): str(row["name"]) for row in cards}
    deck_map_payload = load_json(catalog_dir / "deck_map.json")
    deck_map = deck_map_payload.get("decks", deck_map_payload)
    result: dict[str, tuple[str, ...]] = {}
    for deck_hash, path_text in deck_map.items():
        path = Path(path_text)
        if not path.is_absolute():
            path = catalog_dir / path
        names: list[str] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line in handle:
                # Canonical catalog decks are intentionally headerless: one
                # engine card ID per line, repeated once per physical copy.
                card_id = line.strip().split(",", 1)[0]
                if card_id in card_name:
                    names.append(card_name[card_id])
        result[str(deck_hash)] = normalized_card_names(names)
    return result


def assert_specialist_receipt(receipt: dict[str, Any], threshold: float = 1000.0) -> None:
    if receipt.get("strictPredicate") != STRICT_PREDICATE:
        raise ValueError("specialist receipt predicate is not the strict per-game boundary")
    if float(receipt.get("minScoreExclusive", -1)) != threshold:
        raise ValueError("specialist receipt threshold mismatch")
    if int(receipt.get("episodes", 0)) <= 0:
        raise ValueError("specialist receipt contains no episodes")
    if float(receipt["scoreMin"]) <= threshold:
        raise ValueError("specialist receipt includes score <= 1000")
    if float(receipt["scoreMax"]) < float(receipt["scoreMin"]):
        raise ValueError("specialist receipt score range is invalid")
    if int(receipt.get("duplicateEpisodes", 0)) != 0:
        raise ValueError("specialist receipt contains duplicate episodes")
    if int(receipt.get("trainValidationOverlap", 0)) != 0:
        raise ValueError("specialist receipt train/validation overlap")


def count_catalog(
    catalog_path: Path,
    deck_names: dict[str, tuple[str, ...]],
    config: dict[str, Any],
) -> dict[str, Any]:
    threshold = float(config["minScoreExclusive"])
    archetype_ids = [str(row["id"]) for row in config["archetypes"]]
    totals = {
        name: {
            "episodes": 0,
            "decisions": 0,
            "policyDecisions": 0,
            "winnerPolicyDecisions": 0,
            "lossOrDrawPolicyDecisions": 0,
            "trainEpisodes": 0,
            "validationEpisodes": 0,
            "scoreMin": None,
            "scoreMax": None,
            "deckHashes": set(),
        }
        for name in archetype_ids
    }
    excluded = Counter()
    seen: set[str] = set()
    with catalog_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            episode = str(row["episode_id"])
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
            seat_matches: list[tuple[int, str, str]] = []
            ambiguous = False
            for seat in (0, 1):
                deck_hash = str(row[f"deck{seat}_sha256"])
                matches = matching_archetypes(deck_names.get(deck_hash, ()), config)
                if len(matches) > 1:
                    ambiguous = True
                elif matches:
                    seat_matches.append((seat, matches[0], deck_hash))
            if ambiguous:
                excluded["ambiguousDeck"] += 1
                continue
            if not seat_matches:
                excluded["nonTarget"] += 1
                continue
            for seat, name, deck_hash in seat_matches:
                target = totals[name]
                decisions = int(row[f"nonforced_decisions{seat}"])
                policy_decisions = decisions
                winner_index = int(row["winner_index"])
                winner = winner_index == seat
                target["episodes"] += 1
                target["decisions"] += int(row["nonforced_decisions0"]) + int(row["nonforced_decisions1"])
                target["policyDecisions"] += policy_decisions
                target["winnerPolicyDecisions" if winner else "lossOrDrawPolicyDecisions"] += policy_decisions
                target["validationEpisodes" if validation_episode(episode, config) else "trainEpisodes"] += 1
                target["scoreMin"] = score if target["scoreMin"] is None else min(target["scoreMin"], score)
                target["scoreMax"] = score if target["scoreMax"] is None else max(target["scoreMax"], score)
                target["deckHashes"].add(deck_hash)
    for row in totals.values():
        row["deckHashes"] = sorted(row["deckHashes"])
    return {
        "strictPredicate": STRICT_PREDICATE,
        "minScoreExclusive": threshold,
        "catalog": str(catalog_path),
        "excluded": dict(sorted(excluded.items())),
        "archetypes": totals,
    }
