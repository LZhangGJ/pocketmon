from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_deck(path: Path) -> list[int]:
    deck = [int(value) for value in path.read_text(encoding="utf-8").split()]
    if len(deck) != 60:
        raise ValueError(f"target deck must contain exactly 60 cards, got {len(deck)}")
    return deck


def deck_similarity(left: list[int], right: list[int]) -> float:
    """Multiset overlap divided by the legal 60-card deck size."""

    if len(left) != 60 or len(right) != 60:
        raise ValueError("deck similarity requires two 60-card decks")
    return sum((Counter(left) & Counter(right)).values()) / 60.0


def load_decks(path: Path) -> dict[tuple[str, int], list[int]]:
    decks: dict[tuple[str, int], list[int]] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            episode_id = str(row.get("episode_id"))
            player = row.get("player")
            deck = row.get("deck")
            if player not in (0, 1) or isinstance(player, bool):
                raise ValueError(f"invalid player on deck-map line {line_number}")
            if not isinstance(deck, list) or len(deck) != 60 or any(
                not isinstance(card_id, int) or isinstance(card_id, bool) for card_id in deck
            ):
                raise ValueError(f"invalid deck on deck-map line {line_number}")
            key = (episode_id, int(player))
            if key in decks and decks[key] != deck:
                raise ValueError(f"conflicting deck-map entry for {key}")
            decks[key] = deck
    if not decks:
        raise ValueError("deck map is empty")
    return decks


def filter_replays(
    *,
    input_path: Path,
    deck_map_path: Path,
    target_deck_path: Path,
    output_path: Path,
    audit_path: Path,
    min_similarity: float,
    min_episode_players: int,
    max_episode_players: int = 0,
    selection_seed: int = 20260807,
) -> dict[str, Any]:
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("minimum similarity must be in [0, 1]")
    if min_episode_players <= 0:
        raise ValueError("minimum episode-player count must be positive")
    if max_episode_players < 0:
        raise ValueError("maximum episode-player count must be non-negative")
    if output_path.exists() or audit_path.exists():
        raise FileExistsError("refusing to overwrite specialist replay output")

    target = read_deck(target_deck_path)
    decks = load_decks(deck_map_path)
    similarities = {key: deck_similarity(deck, target) for key, deck in decks.items()}
    eligible = {key for key, similarity in similarities.items() if similarity >= min_similarity}
    if len(eligible) < min_episode_players:
        raise ValueError(
            f"only {len(eligible)} episode-player units matched the specialist deck; "
            f"need at least {min_episode_players}"
        )
    selected = set(eligible)
    if max_episode_players and len(selected) > max_episode_players:
        def selection_key(key: tuple[str, int]) -> str:
            value = f"{selection_seed}:{key[0]}:{key[1]}".encode("utf-8")
            return hashlib.sha256(value).hexdigest()

        selected = set(sorted(selected, key=selection_key)[:max_episode_players])
    if len(selected) < min_episode_players:
        raise ValueError("maximum episode-player cap is below the required minimum")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    counters = Counter()
    selected_episodes: set[str] = set()
    selected_units_seen: set[tuple[str, int]] = set()
    try:
        with gzip.open(input_path, "rt", encoding="utf-8") as source, gzip.open(
            temporary, "wt", encoding="utf-8"
        ) as destination:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                counters["input_rows"] += 1
                row = json.loads(line)
                if row.get("schema_version") != 2:
                    raise ValueError(f"unsupported replay schema on line {line_number}")
                episode_id = str(row.get("episode_id"))
                player = row.get("player")
                if player not in (0, 1) or isinstance(player, bool):
                    raise ValueError(f"invalid replay player on line {line_number}")
                key = (episode_id, int(player))
                if key not in decks:
                    raise ValueError(f"missing submitted deck for replay unit {key}")
                if key not in selected:
                    continue
                destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                counters["output_rows"] += 1
                counters["policy_rows"] += int(float(row.get("policy_weight", 0.0)) == 1.0)
                counters["value_rows"] += int(float(row.get("value_weight", 0.0)) == 1.0)
                selected_episodes.add(episode_id)
                selected_units_seen.add(key)
        if counters["output_rows"] <= 0 or counters["policy_rows"] <= 0:
            raise ValueError("specialist replay filter produced no trainable policy rows")
        temporary.replace(output_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    histogram = Counter(round(similarities[key], 2) for key in selected)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "kind": "deck_specialist_replay_filter",
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "rows": counters["input_rows"],
        },
        "deck_map": {
            "path": str(deck_map_path),
            "sha256": sha256_file(deck_map_path),
            "entries": len(decks),
        },
        "target_deck": {
            "path": str(target_deck_path),
            "sha256": sha256_file(target_deck_path),
            "cards": len(target),
        },
        "selection": {
            "similarity": "multiset_overlap_over_60",
            "min_similarity": min_similarity,
            "eligible_episode_players": len(eligible),
            "selected_episode_players": len(selected),
            "observed_episode_players": len(selected_units_seen),
            "max_episode_players": max_episode_players,
            "selection_seed": selection_seed,
            "selection_uses_outcome_or_action": False,
            "episodes": len(selected_episodes),
            "similarity_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
            "rows": counters["output_rows"],
            "policy_rows": counters["policy_rows"],
            "value_rows": counters["value_rows"],
        },
        "invalid_actions": 0,
        "quarantined_rows": 0,
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter replay rows for one submitted-deck specialist")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--deck-map", required=True, type=Path)
    parser.add_argument("--target-deck", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--min-similarity", type=float, default=0.75)
    parser.add_argument("--min-episode-players", type=int, default=20)
    parser.add_argument("--max-episode-players", type=int, default=0)
    parser.add_argument("--selection-seed", type=int, default=20260807)
    args = parser.parse_args()
    audit = filter_replays(
        input_path=args.input,
        deck_map_path=args.deck_map,
        target_deck_path=args.target_deck,
        output_path=args.output,
        audit_path=args.audit_output,
        min_similarity=args.min_similarity,
        min_episode_players=args.min_episode_players,
        max_episode_players=args.max_episode_players,
        selection_seed=args.selection_seed,
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
