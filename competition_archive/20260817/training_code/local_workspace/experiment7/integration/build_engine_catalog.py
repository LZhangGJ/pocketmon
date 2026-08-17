from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import sha256_file, utc_now, write_json


def load_list(path: Path, expected_key: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict) and isinstance(payload.get(expected_key), list):
        values = payload[expected_key]
    else:
        raise ValueError(f"{path} does not contain a {expected_key} list")
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"{path} contains non-object rows")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine pocketmon card and attack references for Experiment 7")
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--attacks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    cards = load_list(args.cards, "cards")
    attacks = load_list(args.attacks, "attacks")
    card_ids = [int(row["cardId"]) for row in cards]
    attack_ids = [int(row["attackId"]) for row in attacks]
    if len(card_ids) != len(set(card_ids)):
        raise RuntimeError("duplicate card IDs in official card reference")
    if len(attack_ids) != len(set(attack_ids)):
        raise RuntimeError("duplicate attack IDs in official attack reference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"cards": cards, "attacks": attacks}, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    receipt = args.receipt or args.output.with_suffix(".receipt.json")
    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "cards": {"path": str(args.cards.resolve()), "sha256": sha256_file(args.cards), "rows": len(cards)},
        "attacks": {"path": str(args.attacks.resolve()), "sha256": sha256_file(args.attacks), "rows": len(attacks)},
        "output": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output), "bytes": args.output.stat().st_size},
    }
    write_json(receipt, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
