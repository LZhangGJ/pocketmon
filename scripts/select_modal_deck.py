from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import TextIO


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def opener(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def read_deck(path: Path) -> list[int]:
    deck = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(deck) != 60:
        raise ValueError(f"reference deck must contain 60 cards: {path}")
    return deck


def pokemon_ids(card_database: Path) -> set[int]:
    cards = json.loads(card_database.read_text(encoding="utf-8"))
    return {
        int(card["cardId"])
        for card in cards
        if int(card.get("cardType", -1)) == 0
    }


def core_signature(deck: list[int], pokemon: set[int]) -> tuple[tuple[int, int], ...]:
    counts = Counter(card for card in deck if card in pokemon)
    return tuple(sorted(counts.items()))


def deck_sha256(deck: tuple[int, ...]) -> str:
    payload = "\n".join(map(str, deck)).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the modal exact 60-card list among decks sharing the "
            "reference deck's Pokemon core"
        )
    )
    parser.add_argument("--deck-map", type=Path, required=True)
    parser.add_argument("--reference-deck", type=Path, required=True)
    parser.add_argument("--card-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    deck_map = resolve(args.deck_map)
    reference_path = resolve(args.reference_deck)
    card_database = resolve(args.card_database)
    output = resolve(args.output)
    audit_output = resolve(args.audit_output)
    for path in (deck_map, reference_path, card_database):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (output, audit_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite output: {path}")

    reference = read_deck(reference_path)
    pokemon = pokemon_ids(card_database)
    reference_core = core_signature(reference, pokemon)
    exact_counts: Counter[tuple[int, ...]] = Counter()
    matched_entries = 0
    with opener(deck_map) as handle:
        for line_number, line in enumerate(handle, 1):
            raw = json.loads(line)
            deck = raw.get("deck")
            if (
                not isinstance(deck, list)
                or len(deck) != 60
                or any(
                    not isinstance(card, int) or isinstance(card, bool)
                    for card in deck
                )
            ):
                raise ValueError(f"invalid deck-map entry on line {line_number}")
            normalized = tuple(sorted(deck))
            if core_signature(list(normalized), pokemon) != reference_core:
                continue
            exact_counts[normalized] += 1
            matched_entries += 1

    if not exact_counts:
        raise ValueError("no decks match the reference Pokemon core")
    best_count = max(exact_counts.values())
    modal = min(
        deck for deck, count in exact_counts.items()
        if count == best_count
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(f"{card_id}\n" for card_id in modal),
        encoding="utf-8",
    )
    audit = {
        "deck_map": str(deck_map),
        "reference_deck": str(reference_path),
        "reference_core": [list(item) for item in reference_core],
        "matched_deck_entries": matched_entries,
        "unique_exact_decks": len(exact_counts),
        "modal_count": best_count,
        "modal_share_within_core": best_count / matched_entries,
        "modal_deck_sha256": deck_sha256(modal),
        "top_exact_variants": [
            {
                "count": count,
                "share_within_core": count / matched_entries,
                "deck_sha256": deck_sha256(deck),
            }
            for deck, count in sorted(
                exact_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )[:10]
        ],
        "output": str(output),
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
