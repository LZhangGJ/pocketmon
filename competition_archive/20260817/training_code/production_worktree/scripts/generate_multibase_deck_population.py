from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mutate_legal_decks import (  # noqa: E402
    card_index,
    deck_sha,
    is_basic_energy,
    mutate_once,
    pool_frequencies,
    read_deck,
    role,
    validate_deck,
)


@dataclass(frozen=True)
class BaseDeck:
    name: str
    archetype: str
    path: Path
    deck: tuple[int, ...]


def multiset_fingerprint(deck: list[int] | tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    """Identify a deck by card counts; physical card ordering is not gameplay state."""

    return tuple(sorted(Counter(deck).items()))


def weighted_from_counter(rng: random.Random, counts: Counter[int]) -> int:
    ids = sorted(counts)
    return rng.choices(ids, weights=[counts[card_id] for card_id in ids], k=1)[0]


def crossover_once(
    recipient: list[int],
    donor: list[int],
    cards: dict[int, dict[str, Any]],
    rng: random.Random,
    swaps: int,
) -> tuple[list[int], list[dict[str, Any]]] | None:
    """Move same-role cards from a donor archetype into a recipient deck."""

    deck = list(recipient)
    donor_counts = Counter(donor)
    changes: list[dict[str, Any]] = []
    positions = list(range(len(deck)))
    rng.shuffle(positions)
    for index in positions:
        removed_id = deck[index]
        compatible = Counter(
            {
                card_id: count
                for card_id, count in donor_counts.items()
                if card_id != removed_id and role(cards[card_id]) == role(cards[removed_id])
            }
        )
        if not compatible:
            continue
        candidates = list(compatible)
        rng.shuffle(candidates)
        candidates.sort(key=lambda card_id: compatible[card_id], reverse=True)
        added_id = None
        for candidate in candidates:
            trial = list(deck)
            trial[index] = candidate
            if not validate_deck(trial, cards):
                added_id = candidate
                deck = trial
                break
        if added_id is None:
            continue
        changes.append(
            {
                "removedId": removed_id,
                "removedName": cards[removed_id].get("name"),
                "addedId": added_id,
                "addedName": cards[added_id].get("name"),
                "rolePreserved": True,
            }
        )
        if len(changes) == swaps:
            return deck, changes
    return None


def proportion_once(
    base: list[int],
    cards: dict[int, dict[str, Any]],
    frequencies: Counter[int],
    rng: random.Random,
    swaps: int,
) -> tuple[list[int], list[dict[str, Any]]] | None:
    """Adjust card ratios while preserving broad card roles and deck legality."""

    deck = list(base)
    changes: list[dict[str, Any]] = []
    for _ in range(swaps):
        counts = Counter(deck)
        removable = [card_id for card_id, count in counts.items() if count >= 2]
        rng.shuffle(removable)
        replaced = False
        for removed_id in removable:
            compatible = Counter(
                {
                    card_id: 1 + frequencies[card_id] + counts[card_id]
                    for card_id in set(frequencies) | set(counts)
                    if card_id != removed_id
                    and card_id in cards
                    and role(cards[card_id]) == role(cards[removed_id])
                    and (
                        is_basic_energy(cards[card_id])
                        or counts[card_id] < 4
                    )
                }
            )
            if not compatible:
                continue
            for _attempt in range(40):
                added_id = weighted_from_counter(rng, compatible)
                index = deck.index(removed_id)
                trial = list(deck)
                trial[index] = added_id
                if validate_deck(trial, cards):
                    compatible.pop(added_id, None)
                    if not compatible:
                        break
                    continue
                deck = trial
                changes.append(
                    {
                        "removedId": removed_id,
                        "removedName": cards[removed_id].get("name"),
                        "addedId": added_id,
                        "addedName": cards[added_id].get("name"),
                        "rolePreserved": True,
                    }
                )
                replaced = True
                break
            if replaced:
                break
        if not replaced:
            return None
    return deck, changes


def generate_population(
    bases: list[BaseDeck],
    cards: dict[int, dict[str, Any]],
    frequencies: Counter[int],
    count: int,
    seed: int,
    max_swaps: int,
) -> list[dict[str, Any]]:
    if len(bases) < 2:
        raise ValueError("multi-base deck search requires at least two base decks")
    if count < len(bases):
        raise ValueError("candidate count must include every base deck")
    for base in bases:
        errors = validate_deck(list(base.deck), cards)
        if errors:
            raise ValueError(f"illegal base deck {base.name}: {'; '.join(errors)}")

    rng = random.Random(seed)
    combined_frequencies = Counter(frequencies)
    for base in bases:
        combined_frequencies.update(base.deck)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for base in bases:
        fingerprint = multiset_fingerprint(base.deck)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        rows.append(
            {
                "deck": list(base.deck),
                "method": "base",
                "parents": [base.name],
                "archetypeId": base.archetype,
                "archetypeLabel": base.archetype,
                "changes": [],
            }
        )

    methods = ("mutation", "proportion", "crossover")
    attempts = 0
    max_attempts = count * 2000
    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        method = methods[(attempts - 1) % len(methods)]
        base = rng.choice(bases)
        swaps = rng.randint(1, max_swaps)
        if method == "mutation":
            generated = mutate_once(
                list(base.deck), cards, combined_frequencies, rng, swaps
            )
            parents = [base.name]
            archetype = base.archetype
        elif method == "proportion":
            generated = proportion_once(
                list(base.deck), cards, combined_frequencies, rng, swaps
            )
            parents = [base.name]
            archetype = base.archetype
        else:
            donor = rng.choice([candidate for candidate in bases if candidate.name != base.name])
            generated = crossover_once(
                list(base.deck), list(donor.deck), cards, rng, swaps
            )
            parents = [base.name, donor.name]
            archetype = f"{base.archetype}__x__{donor.archetype}"
        if generated is None:
            continue
        deck, changes = generated
        fingerprint = multiset_fingerprint(deck)
        if fingerprint in seen or validate_deck(deck, cards):
            continue
        seen.add(fingerprint)
        rows.append(
            {
                "deck": deck,
                "method": method,
                "parents": parents,
                "archetypeId": archetype,
                "archetypeLabel": archetype,
                "changes": changes,
            }
        )
    if len(rows) != count:
        raise RuntimeError(
            f"generated only {len(rows)}/{count} unique legal deck multisets after {attempts} attempts"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a deterministic legal multi-archetype deck-search population"
    )
    parser.add_argument(
        "--base",
        nargs=3,
        action="append",
        metavar=("NAME", "ARCHETYPE", "DECK"),
        required=True,
    )
    parser.add_argument("--cards", type=Path, default=Path("data/reference/official_cards.json"))
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--max-swaps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite deck population: {output}")
    cards = card_index(args.cards.resolve())
    bases = [
        BaseDeck(name, archetype, Path(path).resolve(), tuple(read_deck(Path(path).resolve())))
        for name, archetype, path in args.base
    ]
    names = [base.name for base in bases]
    if len(names) != len(set(names)):
        raise ValueError("base deck names must be unique")
    frequencies = pool_frequencies(args.pool_manifest.resolve(), cards)
    rows = generate_population(
        bases, cards, frequencies, args.count, args.seed, args.max_swaps
    )

    output.mkdir(parents=True)
    selected = []
    for index, row in enumerate(rows):
        name = f"deck_candidate_{index:03d}"
        deck_path = output / name / "deck.csv"
        deck_path.parent.mkdir()
        deck_path.write_text("".join(f"{card_id}\n" for card_id in row["deck"]), encoding="utf-8")
        selected.append(
            {
                "name": name,
                "deckPath": str(deck_path),
                "deckSha256": deck_sha(row["deck"]),
                "archetypeId": row["archetypeId"],
                "archetypeLabel": row["archetypeLabel"],
                "method": row["method"],
                "parents": row["parents"],
                "changes": row["changes"],
                "legal": True,
            }
        )
    manifest = {
        "schemaVersion": 1,
        "kind": "experiment7_multibase_deck_search_population",
        "seed": args.seed,
        "maxSwaps": args.max_swaps,
        "candidateCount": len(selected),
        "cards": str(args.cards.resolve()),
        "poolManifest": str(args.pool_manifest.resolve()),
        "bases": [
            {
                "name": base.name,
                "archetype": base.archetype,
                "deckPath": str(base.path),
                "deckSha256": deck_sha(list(base.deck)),
            }
            for base in bases
        ],
        "methodCounts": dict(Counter(row["method"] for row in rows)),
        "selected": selected,
    }
    manifest_path = output / "population.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    print(json.dumps({"manifest": str(manifest_path), "sha256": digest, **manifest["methodCounts"]}))


if __name__ == "__main__":
    main()
