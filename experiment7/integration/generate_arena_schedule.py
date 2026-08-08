from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic seat-balanced challenger-vs-Lucario schedules")
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--target-agent", type=Path, required=True)
    parser.add_argument("--games-per-challenger", type=int, required=True)
    parser.add_argument("--seed-base", type=int, default=2026080800)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.games_per_challenger <= 0 or args.games_per_challenger % 2:
        raise ValueError("games-per-challenger must be a positive even number")
    payload = json.loads(args.package_manifest.read_text(encoding="utf-8"))
    packages = payload.get("packages") or []
    if not packages:
        raise ValueError("package manifest is empty")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    learners = {
        "agents": [
            {
                "name": row["deck_name"],
                "agent_dir": row["package"],
                "status": "accepted",
            }
            for row in packages
        ]
    }
    opponents = {
        "agents": [
            {
                "name": "lucario_rule_frozen",
                "agent_dir": str(args.target_agent.resolve()),
                "status": "accepted",
            }
        ]
    }
    write_json(args.output_dir / "learners.json", learners)
    write_json(args.output_dir / "opponents.json", opponents)
    fields = ["learner", "opponent", "seed", "learner_seat"]
    with (args.output_dir / "schedule.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        game_id = 0
        half = args.games_per_challenger // 2
        for package in packages:
            for seat in (0, 1):
                for local in range(half):
                    writer.writerow(
                        {
                            "learner": package["deck_name"],
                            "opponent": "lucario_rule_frozen",
                            "seed": args.seed_base + game_id,
                            "learner_seat": seat,
                        }
                    )
                    game_id += 1
    write_json(
        args.output_dir / "schedule_receipt.json",
        {
            "schema_version": 1,
            "challengers": len(packages),
            "games_per_challenger": args.games_per_challenger,
            "total_games": game_id,
            "seat_balanced": True,
            "target_agent": str(args.target_agent.resolve()),
            "seed_base": args.seed_base,
        },
    )
    print(args.output_dir / "schedule.csv")


if __name__ == "__main__":
    main()
