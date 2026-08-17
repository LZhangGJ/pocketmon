#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


UNFAIR_STAMP = 1080
MAXIMUM_BELT = 1158
GOLDEEN = 100
SEAKING = 240
PETILIL = 321
LILLIGANT = 322


def read_deck(path: Path) -> list[int]:
    cards = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"expected 60 cards in {path}, got {len(cards)}")
    return cards


def replace_exact(cards: list[int], remove: Counter[int], add: list[int]) -> list[int]:
    result = list(cards)
    for card_id, count in remove.items():
        if result.count(card_id) != count:
            raise ValueError(
                f"expected exactly {count} copies of {card_id}, found {result.count(card_id)}"
            )
        for _ in range(count):
            result.remove(card_id)
    result.extend(add)
    result.sort()
    if len(result) != 60:
        raise ValueError(f"variant has {len(result)} cards")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-deck", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    base = read_deck(args.base_deck.resolve())
    variants = {
        "a08_maxbelt": replace_exact(base, Counter({UNFAIR_STAMP: 1}), [MAXIMUM_BELT]),
        "a08_lilligant": replace_exact(
            base,
            Counter({GOLDEEN: 2, SEAKING: 2}),
            [PETILIL, PETILIL, LILLIGANT, LILLIGANT],
        ),
        "a08_lilligant_maxbelt": replace_exact(
            replace_exact(
                base,
                Counter({GOLDEEN: 2, SEAKING: 2}),
                [PETILIL, PETILIL, LILLIGANT, LILLIGANT],
            ),
            Counter({UNFAIR_STAMP: 1}),
            [MAXIMUM_BELT],
        ),
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, cards in variants.items():
        path = args.output_root / f"{name}.csv"
        if path.exists():
            existing = read_deck(path)
            if existing != cards:
                raise FileExistsError(f"refusing to overwrite different deck: {path}")
        else:
            path.write_text("".join(f"{card}\n" for card in cards), encoding="utf-8")
        rows.append(
            {
                "name": name,
                "deckPath": str(path.resolve()),
                "cards": len(cards),
                "counts": {str(key): value for key, value in sorted(Counter(cards).items())},
            }
        )

    manifest = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "baseDeck": str(args.base_deck.resolve()),
        "variants": rows,
        "evaluationContract": {
            "archaludonGames": 80,
            "archaludonSeatBalanced": True,
            "frozenGamesPerAgent": 20,
            "directUniversalBcGames": 40,
            "promotionArchaludonScoreRate": 0.45,
            "maximumFrozenRegressionPp": 2.0,
            "requiredConsecutivePasses": 2,
        },
    }
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous.pop("createdAt", None)
        comparable = dict(manifest)
        comparable.pop("createdAt", None)
        if previous != comparable:
            raise FileExistsError(f"refusing to overwrite different manifest: {manifest_path}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
