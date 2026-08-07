from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_decks(replay: dict[str, Any], source: Path) -> list[list[int]]:
    steps = replay.get("steps")
    if not isinstance(steps, list) or len(steps) < 2 or not isinstance(steps[1], list):
        raise ValueError(f"{source}: missing initial deck action step")
    decks: list[list[int]] = []
    for player in range(2):
        if player >= len(steps[1]) or not isinstance(steps[1][player], dict):
            raise ValueError(f"{source}: missing player {player} deck action")
        deck = steps[1][player].get("action")
        if not isinstance(deck, list) or len(deck) != 60:
            raise ValueError(f"{source}: player {player} deck must contain 60 cards")
        if any(not isinstance(card_id, int) or isinstance(card_id, bool) or card_id <= 0 for card_id in deck):
            raise ValueError(f"{source}: player {player} deck contains an invalid card ID")
        decks.append(deck)
    return decks


DECK_ACTION = re.compile(
    rb'"action"\s*:\s*\[\s*\[([0-9,\s]+)\]\s*,\s*\[([0-9,\s]+)\]\s*\]'
)


def extract_decks_from_payload(payload: bytes, source: Path) -> list[list[int]]:
    """Use the initial visualize action fast path; fall back to full JSON for diagnostics."""
    match = DECK_ACTION.search(payload)
    if match:
        decks = [
            [int(value) for value in group.split(b",") if value.strip()]
            for group in match.groups()
        ]
        if len(decks) == 2 and all(len(deck) == 60 and all(card_id > 0 for card_id in deck) for deck in decks):
            return decks
    return extract_decks(json.loads(payload), source)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build acting-player deck sidecar for structured replay training")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    args = parser.parse_args()

    raw_dir, output = Path(args.raw_dir), Path(args.output)
    paths = sorted(raw_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no replay JSON files under {raw_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    episodes = entries = 0
    with gzip.open(output, "wt", encoding="utf-8", newline="\n") as handle:
        for index, path in enumerate(paths, 1):
            payload = path.read_bytes()
            decks = extract_decks_from_payload(payload, path)
            source_sha256 = hashlib.sha256(payload).hexdigest()
            for player, deck in enumerate(decks):
                record = {
                    "episode_id": path.stem,
                    "player": player,
                    "deck": deck,
                    "source_sha256": source_sha256,
                }
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                entries += 1
            episodes += 1
            if index % 250 == 0:
                print(json.dumps({"processed": index, "total": len(paths)}), flush=True)
    metadata = {
        "raw_dir": str(raw_dir),
        "raw_files": len(paths),
        "episodes": episodes,
        "entries": entries,
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256_file(output),
        "deck_size": 60,
        "scope": "acting player's own submitted deck only",
    }
    metadata_path = Path(args.metadata_output)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
