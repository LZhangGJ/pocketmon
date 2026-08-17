#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    league = load(args.league_root / "state/league.json")
    pool = load(Path(league["basePool"]["path"]))
    by_name = {row["name"]: row for row in pool["agents"]}
    submission = dict(by_name["team_submission_4_portable_bc"])
    submission["name"] = "submission4_full_pool_candidate"
    champion = by_name["champion_a05_raging_bolt_ogerpon_kangaskhan_g000080"]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / "learners.json", {"schemaVersion": 1, "agents": [submission]})
    write(output / "opponents.json", {"schemaVersion": 1, "agents": [champion]})
    with (output / "schedule.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("learner", "opponent", "seed", "learner_seat"))
        writer.writeheader()
        for index in range(2):
            seed = 263_811_900 + index
            writer.writerow({"learner": submission["name"], "opponent": champion["name"], "seed": seed, "learner_seat": 0})
            writer.writerow({"learner": submission["name"], "opponent": champion["name"], "seed": seed, "learner_seat": 1})
    write(
        output / "metadata.json",
        {
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "engineSeedControlled": False,
            "purpose": "complete submission4 coverage over all 26 non-self frozen agents",
        },
    )
    print(output)


if __name__ == "__main__":
    main()
