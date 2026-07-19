from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Matchup:
    learner: str
    opponent: str
    seed: int
    learner_seat: int


def wilson_lower(wins: int, games: int, z: float = 1.959963984540054) -> float:
    if games <= 0:
        return 0.0
    p = wins / games
    d = 1 + z * z / games
    centre = p + z * z / (2 * games)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * games)) / games)
    return (centre - margin) / d


def build_schedule(learners: int, opponents: list[str], games: int, seed: int) -> list[Matchup]:
    if learners != 10:
        raise ValueError("League-v1 requires exactly 10 learners")
    if games < 2 or games % 2:
        raise ValueError("games_per_pair must be a positive even number")
    rows = []
    for learner_index in range(learners):
        learner = f"learner_{learner_index:02d}"
        for opponent_index, opponent in enumerate(opponents):
            pair_seed = seed + learner_index * 1_000_000 + opponent_index * 10_000
            for pair in range(games // 2):
                game_seed = pair_seed + pair
                rows.append(Matchup(learner, opponent, game_seed, 0))
                rows.append(Matchup(learner, opponent, game_seed, 1))
    random.Random(seed).shuffle(rows)
    return rows


def evaluate(rows: list[dict], config: dict) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["learner"], row["opponent"]), []).append(row)
    thresholds = config["evaluation"]
    by_learner: dict[str, dict] = {}
    for (learner, opponent), games in sorted(grouped.items()):
        wins = sum(g["result"] == "win" for g in games)
        failures = sum(g["result"] in {"crash", "timeout", "illegal"} for g in games)
        rate = wins / len(games)
        lower = wilson_lower(wins, len(games))
        passed = (
            rate >= thresholds["min_public_win_rate"]
            and lower >= thresholds["min_lower_wilson_bound"]
            and failures / len(games) <= thresholds["max_failure_rate"]
        )
        by_learner.setdefault(learner, {"matchups": {}, "qualified": True})
        by_learner[learner]["matchups"][opponent] = {
            "games": len(games), "wins": wins, "win_rate": rate,
            "wilson_lower": lower, "failures": failures, "passed": passed,
        }
        by_learner[learner]["qualified"] &= passed
    qualified = sorted(k for k, v in by_learner.items() if v["qualified"])
    required = config["promotion"]["required_qualified_learners"]
    return {"learners": by_learner, "qualified": qualified, "promote": len(qualified) >= required}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or score the reproducible League-v1 gate")
    parser.add_argument("--config", default="configs/league_v1.json")
    parser.add_argument("--snapshot")
    parser.add_argument("--schedule")
    parser.add_argument("--results")
    parser.add_argument("--report")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.results:
        with Path(args.results).open(newline="", encoding="utf-8") as handle:
            report = evaluate(list(csv.DictReader(handle)), config)
        output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.report:
            Path(args.report).write_text(output, encoding="utf-8")
        print(output, end="")
        return
    snapshot_path = Path(args.snapshot or config["opponent_snapshot"])
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    opponents = [item["name"] for item in snapshot["agents"] if item["status"] == "accepted"]
    schedule = build_schedule(config["learners"], opponents, config["evaluation"]["games_per_pair"], config["base_seed"])
    output = Path(args.schedule or "results/league_v1_schedule.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Matchup.__annotations__))
        writer.writeheader()
        writer.writerows(row.__dict__ for row in schedule)
    print(json.dumps({"schedule": str(output), "games": len(schedule), "opponents": len(opponents)}))


if __name__ == "__main__":
    main()
