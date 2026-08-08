from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine pocketmon card and attack references for Experiment 7")
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--attacks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    attacks = json.loads(args.attacks.read_text(encoding="utf-8"))
    if not isinstance(cards, list) or not cards:
        raise ValueError("cards reference must be a non-empty JSON list")
    if not isinstance(attacks, list) or not attacks:
        raise ValueError("attacks reference must be a non-empty JSON list")
    card_ids = [int(row["cardId"]) for row in cards]
    attack_ids = [int(row["attackId"]) for row in attacks]
    if len(card_ids) != len(set(card_ids)) or min(card_ids) <= 0:
        raise ValueError("card IDs must be unique positive integers")
    if len(attack_ids) != len(set(attack_ids)) or min(attack_ids) <= 0:
        raise ValueError("attack IDs must be unique positive integers")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"cards": cards, "attacks": attacks}, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    write_json(
        args.output.with_suffix(args.output.suffix + ".receipt.json"),
        {
            "schema_version": 1,
            "output": str(args.output.resolve()),
            "output_sha256": sha256_file(args.output),
            "cards": {"path": str(args.cards.resolve()), "sha256": sha256_file(args.cards), "count": len(cards)},
            "attacks": {"path": str(args.attacks.resolve()), "sha256": sha256_file(args.attacks), "count": len(attacks)},
        },
    )
    print(args.output)


if __name__ == "__main__":
    main()
