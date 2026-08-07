from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def opener(path: Path, mode: str):
    return gzip.open(path, mode, encoding="utf-8") if path.suffix == ".gz" else path.open(mode, encoding="utf-8")


def read_deck(path: Path) -> list[int]:
    cards = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"target deck must contain exactly 60 card IDs: {path} has {len(cards)}")
    return cards


def pokemon_ids(card_database: Path) -> set[int]:
    cards = json.loads(card_database.read_text(encoding="utf-8"))
    return {int(card["cardId"]) for card in cards if int(card["cardType"]) == 0}


def signature(deck: list[int], mode: str, pokemon: set[int]) -> tuple[tuple[int, int], ...]:
    counts = Counter(deck)
    if mode == "pokemon-core":
        counts = Counter({card_id: count for card_id, count in counts.items() if card_id in pokemon})
    return tuple(sorted(counts.items()))


def load_matching_keys(
    deck_map_path: Path,
    target_deck: list[int],
    mode: str,
    pokemon: set[int],
) -> tuple[dict[tuple[str, int], list[int]], set[tuple[str, int]]]:
    decks: dict[tuple[str, int], list[int]] = {}
    target_signature = signature(target_deck, mode, pokemon)
    matching: set[tuple[str, int]] = set()
    with opener(deck_map_path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = json.loads(line)
            key = (str(raw["episode_id"]), int(raw["player"]))
            deck = [int(card_id) for card_id in raw["deck"]]
            if len(deck) != 60:
                raise ValueError(f"invalid deck map entry on line {line_number}")
            decks[key] = deck
            if signature(deck, mode, pokemon) == target_signature:
                matching.add(key)
    if not matching:
        raise ValueError(f"target deck has no {mode} matches in {deck_map_path}")
    return decks, matching


def row_key(raw: dict[str, Any]) -> tuple[str, int]:
    player = raw.get("player")
    if not isinstance(player, int) or isinstance(player, bool):
        raise ValueError(f"invalid player in replay row: {player!r}")
    return str(raw["episode_id"]), player


def stable_key(seed: int, key: tuple[str, int]) -> str:
    return hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode("utf-8")).hexdigest()


def select_generic_keys(
    input_path: Path,
    matching: set[tuple[str, int]],
    generic_fraction: float,
    seed: int,
) -> tuple[set[tuple[str, int]], dict[str, int]]:
    target_policy = target_rows = 0
    generic_policy_by_key: dict[tuple[str, int], int] = defaultdict(int)
    generic_rows_by_key: dict[tuple[str, int], int] = defaultdict(int)
    with opener(input_path, "rt") as handle:
        for line in handle:
            raw = json.loads(line)
            key = row_key(raw)
            if key in matching:
                target_rows += 1
                target_policy += int(float(raw.get("policy_weight", 0.0)) == 1.0)
            else:
                generic_rows_by_key[key] += 1
                generic_policy_by_key[key] += int(float(raw.get("policy_weight", 0.0)) == 1.0)
    if target_policy == 0:
        raise ValueError("matched target deck has zero policy rows")
    desired_generic_policy = round(target_policy * generic_fraction / (1.0 - generic_fraction))
    selected: set[tuple[str, int]] = set()
    selected_policy = 0
    selected_rows = 0
    candidates = sorted(generic_policy_by_key, key=lambda key: stable_key(seed, key))
    for key in candidates:
        if selected_policy >= desired_generic_policy:
            break
        if generic_policy_by_key[key] == 0:
            continue
        selected.add(key)
        selected_policy += generic_policy_by_key[key]
        selected_rows += generic_rows_by_key[key]
    return selected, {
        "target_rows": target_rows,
        "target_policy_rows": target_policy,
        "desired_generic_policy_rows": desired_generic_policy,
        "selected_generic_rows": selected_rows,
        "selected_generic_policy_rows": selected_policy,
    }


def write_outputs(
    input_path: Path,
    deck_map_path: Path,
    output_path: Path,
    output_deck_map: Path,
    selected_keys: set[tuple[str, int]],
    decks: dict[tuple[str, int], list[int]],
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_deck_map.parent.mkdir(parents=True, exist_ok=True)
    rows = policy_rows = value_rows = 0
    with opener(input_path, "rt") as source, opener(output_path, "wt") as target:
        for line in source:
            raw = json.loads(line)
            if row_key(raw) not in selected_keys:
                continue
            target.write(json.dumps(raw, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows += 1
            policy_rows += int(float(raw.get("policy_weight", 0.0)) == 1.0)
            value_rows += int(float(raw.get("value_weight", 0.0)) == 1.0)
    with opener(output_deck_map, "wt") as target:
        for episode_id, player in sorted(selected_keys):
            deck = decks[(episode_id, player)]
            target.write(json.dumps({"episode_id": episode_id, "player": player, "deck": deck}, separators=(",", ":")) + "\n")
    return {"rows": rows, "policy_rows": policy_rows, "value_rows": value_rows, "deck_entries": len(selected_keys)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic deck-specialist replay dataset")
    parser.add_argument("--input", default="data/processed/public_replay_2026-08-05.jsonl.gz")
    parser.add_argument("--deck-map", default="data/processed/replay_decks_2026-08-05.jsonl.gz")
    parser.add_argument("--target-deck", required=True)
    parser.add_argument("--match", choices=("exact", "pokemon-core"), default="pokemon-core")
    parser.add_argument("--generic-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--card-database", default="data/reference/official_cards.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-deck-map", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.generic_fraction < 1.0:
        raise ValueError("--generic-fraction must be in [0, 1)")

    input_path = resolve(args.input)
    deck_map_path = resolve(args.deck_map)
    target_deck_path = resolve(args.target_deck)
    output_path = resolve(args.output)
    output_deck_map = resolve(args.output_deck_map)
    audit_output = resolve(args.audit_output)
    pokemon = pokemon_ids(resolve(args.card_database))
    target_deck = read_deck(target_deck_path)
    decks, matching = load_matching_keys(deck_map_path, target_deck, args.match, pokemon)
    generic, selection_audit = select_generic_keys(
        input_path, matching, args.generic_fraction, args.seed
    )
    selected = matching | generic
    written = write_outputs(input_path, deck_map_path, output_path, output_deck_map, selected, decks)
    audit = {
        "input": str(input_path),
        "deck_map": str(deck_map_path),
        "target_deck": str(target_deck_path),
        "match": args.match,
        "matching_deck_entries": len(matching),
        "generic_fraction_requested": args.generic_fraction,
        "seed": args.seed,
        "selection": selection_audit,
        "output": str(output_path),
        "output_deck_map": str(output_deck_map),
        "written": written,
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
