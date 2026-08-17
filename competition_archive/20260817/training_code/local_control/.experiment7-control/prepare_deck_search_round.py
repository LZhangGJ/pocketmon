from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a frozen 20-game deck-search Arena round")
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--opponents", type=Path, required=True)
    parser.add_argument("--opponent", action="append", required=True)
    parser.add_argument("--games-per-pair", type=int, default=20)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite deck-search round: {output}")
    if args.games_per_pair < 2 or args.games_per_pair % 2:
        raise ValueError("games per pair must be a positive even number")
    packages_path = args.packages.resolve()
    opponents_path = args.opponents.resolve()
    packages = load_json(packages_path).get("packages", [])
    if not packages:
        raise ValueError("package manifest contains no packages")
    learner_names = [str(row["name"]) for row in packages]
    if len(learner_names) != len(set(learner_names)):
        raise ValueError("package names must be unique")
    learner_items = [
        {
            "name": row["name"],
            "agent_dir": row["agentDir"],
            "directorySha256": row["directorySha256"],
            "deckSha256": row["deckSha256"],
            "archetypeId": row.get("archetypeId"),
            "archetypeLabel": row.get("archetypeLabel"),
            "status": "accepted",
        }
        for row in packages
    ]

    opponent_payload = load_json(opponents_path)
    all_opponents = {
        str(row["name"]): row
        for row in opponent_payload.get("agents", [])
        if row.get("status", "accepted") == "accepted"
    }
    opponent_names = list(args.opponent)
    if len(opponent_names) != len(set(opponent_names)):
        raise ValueError("selected opponents must be unique")
    missing = sorted(set(opponent_names) - set(all_opponents))
    if missing:
        raise ValueError(f"selected opponents are absent from pool: {missing}")
    opponent_items = [all_opponents[name] for name in opponent_names]

    schedule = []
    for learner_index, learner in enumerate(learner_names):
        for opponent_index, opponent in enumerate(opponent_names):
            pair_seed = args.seed + learner_index * 1_000_000 + opponent_index * 10_000
            for pair in range(args.games_per_pair // 2):
                game_seed = pair_seed + pair
                schedule.append({"learner": learner, "opponent": opponent, "seed": game_seed, "learner_seat": 0})
                schedule.append({"learner": learner, "opponent": opponent, "seed": game_seed, "learner_seat": 1})
    random.Random(args.seed).shuffle(schedule)

    output.mkdir(parents=True)
    learners_out = output / "learners.json"
    opponents_out = output / "opponents.json"
    schedule_out = output / "schedule20.csv"
    learners_out.write_text(
        json.dumps({"schemaVersion": 1, "agents": learner_items}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    opponents_out.write_text(
        json.dumps({**{key: value for key, value in opponent_payload.items() if key != "agents"}, "agents": opponent_items}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with schedule_out.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("learner", "opponent", "seed", "learner_seat"))
        writer.writeheader()
        writer.writerows(schedule)
    receipt = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "learners": len(learner_items),
        "opponents": opponent_names,
        "gamesPerPair": args.games_per_pair,
        "games": len(schedule),
        "sources": {
            "packages": {"path": str(packages_path), "sha256": sha256_file(packages_path)},
            "opponents": {"path": str(opponents_path), "sha256": sha256_file(opponents_path)},
        },
        "outputs": {
            "learners": {"path": str(learners_out), "sha256": sha256_file(learners_out)},
            "opponents": {"path": str(opponents_out), "sha256": sha256_file(opponents_out)},
            "schedule": {"path": str(schedule_out), "sha256": sha256_file(schedule_out)},
        },
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"learners": len(learner_items), "opponents": len(opponent_items), "games": len(schedule)}))


if __name__ == "__main__":
    main()
