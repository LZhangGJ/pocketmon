from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_deck(path: Path) -> list[int]:
    deck = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"prototype deck must contain 60 cards: {path}")
    return deck


def freeze(pool_path: Path) -> dict[str, Any]:
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    items = payload.get("agents", []) if isinstance(payload, dict) else payload
    prototypes = []
    seen = set()
    for item in items:
        if item.get("status", "accepted") != "accepted":
            continue
        agent_dir = Path(item.get("agent_dir") or item.get("path") or "")
        deck_path = agent_dir / "deck.csv"
        deck = read_deck(deck_path)
        fingerprint = hashlib.sha256(
            ",".join(str(card_id) for card_id in sorted(deck)).encode("ascii")
        ).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        prototypes.append({
            "name": str(item["name"]),
            "deck": sorted(deck),
            "prior": 1.0,
            "deck_sha256": sha256(deck_path),
            "source_agent_dir": str(agent_dir),
        })
    if not prototypes:
        raise ValueError("opponent pool produced no unique deck prototypes")
    return {
        "schema_version": 1,
        "method": "frozen_public_visible_multiset_belief",
        "pool": str(pool_path),
        "pool_sha256": sha256(pool_path),
        "uses_outcomes": False,
        "uses_hidden_runtime_cards": False,
        "prototypes": prototypes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze public opponent decks for leakage-safe belief inference")
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = freeze(args.pool.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "prototypes": len(result["prototypes"])}))


if __name__ == "__main__":
    main()
