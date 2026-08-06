from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_deck(path: Path) -> list[int]:
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def deck_sha(deck: list[int]) -> str:
    return hashlib.sha256(("\n".join(map(str, deck)) + "\n").encode()).hexdigest()


def card_index(path: Path) -> dict[int, dict[str, Any]]:
    return {int(card["cardId"]): card for card in json.loads(path.read_text(encoding="utf-8"))}


def is_basic_energy(card: dict[str, Any]) -> bool:
    return int(card.get("cardType", -1)) == 5 and str(card.get("name", "")).lower().startswith("basic ")


def validate_deck(deck: list[int], cards: dict[int, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(deck) != 60:
        errors.append(f"deck has {len(deck)} cards, expected 60")
    missing = sorted(set(deck) - cards.keys())
    if missing:
        errors.append(f"unknown card IDs: {missing}")
        return errors
    by_name = Counter(cards[card_id].get("name", f"id:{card_id}") for card_id in deck)
    for name, count in sorted(by_name.items()):
        examples = [cards[card_id] for card_id in deck if cards[card_id].get("name") == name]
        if count > 4 and not all(is_basic_energy(card) for card in examples):
            errors.append(f"more than four copies by card name: {name} x{count}")
    ace_specs = sum(bool(cards[card_id].get("aceSpec")) for card_id in deck)
    if ace_specs > 1:
        errors.append(f"more than one ACE SPEC: {ace_specs}")
    basic_pokemon = sum(
        int(cards[card_id].get("cardType", -1)) == 0 and bool(cards[card_id].get("basic"))
        for card_id in deck
    )
    if basic_pokemon == 0:
        errors.append("deck has no Basic Pokemon")
    return errors


def role(card: dict[str, Any]) -> tuple[Any, ...]:
    card_type = int(card.get("cardType", -1))
    if card_type == 0:
        return (card_type, bool(card.get("basic")), bool(card.get("stage1")), bool(card.get("stage2")))
    return (card_type,)


def pool_frequencies(manifest: Path, cards: dict[int, dict[str, Any]]) -> Counter[int]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    items = payload.get("agents", []) if isinstance(payload, dict) else payload
    result: Counter[int] = Counter()
    for item in items:
        if item.get("status", "accepted") != "accepted":
            continue
        path = Path(item.get("agent_dir") or item.get("path", "")) / "deck.csv"
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            continue
        result.update(card_id for card_id in read_deck(path) if card_id in cards)
    return result


def weighted_choice(rng: random.Random, ids: list[int], frequencies: Counter[int]) -> int:
    weights = [1.0 + frequencies[card_id] for card_id in ids]
    return rng.choices(ids, weights=weights, k=1)[0]


def mutate_once(
    base: list[int],
    cards: dict[int, dict[str, Any]],
    frequencies: Counter[int],
    rng: random.Random,
    swaps: int,
) -> tuple[list[int], list[dict[str, Any]]] | None:
    deck = list(base)
    changes: list[dict[str, Any]] = []
    all_pool = sorted(frequencies) or sorted(cards)
    for _ in range(swaps):
        remove_index = rng.randrange(len(deck))
        removed_id = deck.pop(remove_index)
        removed = cards[removed_id]
        same_role = [card_id for card_id in all_pool if role(cards[card_id]) == role(removed)]
        candidate_pool = same_role if same_role and rng.random() < 0.85 else all_pool
        added = None
        for _attempt in range(100):
            candidate = weighted_choice(rng, candidate_pool, frequencies)
            trial = deck + [candidate]
            if not validate_deck(trial, cards):
                added = candidate
                break
        if added is None:
            return None
        deck.insert(remove_index, added)
        changes.append({
            "removed_id": removed_id,
            "removed_name": removed.get("name"),
            "added_id": added,
            "added_name": cards[added].get("name"),
            "role_preserved": role(removed) == role(cards[added]),
        })
    return deck, changes


def generate(
    base: list[int],
    cards: dict[int, dict[str, Any]],
    frequencies: Counter[int],
    count: int,
    seed: int,
    max_swaps: int,
) -> list[dict[str, Any]]:
    errors = validate_deck(base, cards)
    if errors:
        raise ValueError("illegal base deck: " + "; ".join(errors))
    rng = random.Random(seed)
    seen = {tuple(base)}
    result: list[dict[str, Any]] = []
    attempts = 0
    while len(result) < count and attempts < count * 500:
        attempts += 1
        swaps = rng.randint(1, max_swaps)
        candidate = mutate_once(base, cards, frequencies, rng, swaps)
        if candidate is None:
            continue
        deck, changes = candidate
        key = tuple(deck)
        if key in seen or validate_deck(deck, cards):
            continue
        seen.add(key)
        result.append({"deck": deck, "changes": changes, "sha256": deck_sha(deck)})
    if len(result) != count:
        raise RuntimeError(f"generated only {len(result)}/{count} unique legal decks after {attempts} attempts")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic 1-4 card legal deck mutations for League screening")
    parser.add_argument("--base-deck", required=True, type=Path)
    parser.add_argument("--cards", default="data/reference/official_cards.json", type=Path)
    parser.add_argument("--pool-manifest", default="configs/opponent_pool.json", type=Path)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--max-swaps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cards = card_index(args.cards)
    base = read_deck(args.base_deck)
    frequencies = pool_frequencies(args.pool_manifest, cards)
    candidates = generate(base, cards, frequencies, args.count, args.seed, args.max_swaps)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "base_deck": str(args.base_deck),
        "base_sha256": deck_sha(base),
        "seed": args.seed,
        "max_swaps": args.max_swaps,
        "candidate_count": len(candidates),
        "candidate_pool_cards": len(frequencies),
        "candidates": [],
    }
    for index, candidate in enumerate(candidates):
        name = f"candidate_{index:03d}"
        path = args.output / name / "deck.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(map(str, candidate["deck"])) + "\n", encoding="utf-8")
        manifest["candidates"].append({
            "name": name,
            "deck": str(path),
            "sha256": candidate["sha256"],
            "changes": candidate["changes"],
            "legal": True,
        })
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "candidates": len(candidates), "pool_cards": len(frequencies)}))


if __name__ == "__main__":
    main()
