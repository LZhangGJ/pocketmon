from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def extract_modal_deck(deck_map: Path) -> tuple[list[int], int, int]:
    decks: Counter[tuple[int, ...]] = Counter()
    with gzip.open(deck_map, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            deck = raw.get("deck")
            if not isinstance(deck, list) or len(deck) != 60 or not all(
                isinstance(card_id, int) and not isinstance(card_id, bool) for card_id in deck
            ):
                raise ValueError("deck map contains an invalid deck")
            decks[tuple(deck)] += 1
    if not decks:
        raise ValueError("deck map is empty")
    modal, count = decks.most_common(1)[0]
    return list(modal), count, len(decks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover the unique 60-card deck used by a specialist replay set")
    parser.add_argument("--deck-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deck, support, distinct_decks = extract_modal_deck(args.deck_map)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{card_id}\n" for card_id in deck), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "card_count": len(deck),
        "modal_support": support,
        "distinct_decks": distinct_decks,
    }))


if __name__ == "__main__":
    main()
