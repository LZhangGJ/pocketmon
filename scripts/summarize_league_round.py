from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


FAILURES = {"crash", "timeout", "illegal"}


def wilson_lower(wins: int, games: int, z: float = 1.959963984540054) -> float:
    if games <= 0:
        return 0.0
    p = wins / games
    denominator = 1 + z * z / games
    centre = p + z * z / (2 * games)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * games)) / games)
    return (centre - margin) / denominator


def key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return row["learner"], row["opponent"], int(row["seed"]), int(row["learner_seat"])


def load_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row_key = key(row)
                if row_key in seen:
                    raise ValueError(f"duplicate League result: {row_key}")
                seen.add(row_key)
                rows.append(row)
    return rows


def metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    games = len(rows)
    wins = sum(row["result"] == "win" for row in rows)
    losses = sum(row["result"] == "loss" for row in rows)
    draws = sum(row["result"] == "draw" for row in rows)
    failures = sum(row["result"] in FAILURES for row in rows)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "failures": failures,
        "win_rate": wins / games if games else 0.0,
        "score_rate": (wins + 0.5 * draws) / games if games else 0.0,
        "win_wilson_lower": wilson_lower(wins, games),
        "failure_rate": failures / games if games else 0.0,
    }


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    paired: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    by_learner: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_matchup: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        paired[(row["learner"], row["opponent"], int(row["seed"]))].add(int(row["learner_seat"]))
        by_learner[row["learner"]].append(row)
        by_matchup[(row["learner"], row["opponent"])].append(row)
    incomplete_pairs = [(*pair, sorted(seats)) for pair, seats in paired.items() if seats != {0, 1}]
    learners = []
    matchups: dict[str, dict[str, Any]] = {}
    for learner, learner_rows in sorted(by_learner.items()):
        record = {"learner": learner, **metrics(learner_rows)}
        seat_metrics = {
            str(seat): metrics([row for row in learner_rows if int(row["learner_seat"]) == seat])
            for seat in (0, 1)
        }
        learner_matchups = {
            opponent: metrics(by_matchup[(learner, opponent)])
            for candidate, opponent in sorted(by_matchup)
            if candidate == learner
        }
        worst = min(
            learner_matchups,
            key=lambda opponent: (learner_matchups[opponent]["score_rate"], opponent),
        )
        record["seat_metrics"] = seat_metrics
        record["worst_opponent"] = worst
        record["worst_score_rate"] = learner_matchups[worst]["score_rate"]
        learners.append(record)
        matchups[learner] = learner_matchups
    learners.sort(
        key=lambda row: (row["failure_rate"] == 0, row["score_rate"], row["win_wilson_lower"]),
        reverse=True,
    )
    for rank, row in enumerate(learners, start=1):
        row["rank"] = rank
    return {
        "games": len(rows),
        "learners": learners,
        "matchups": matchups,
        "paired_seat_groups": len(paired),
        "incomplete_pairs": incomplete_pairs,
        "engine_seed_controlled": all(row.get("engine_seed_controlled", "").lower() == "true" for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and rank a completed paired-seat League screen")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    rows = load_rows(args.inputs)
    report = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "merged_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "ranking.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "rank", "learner", "games", "wins", "losses", "draws", "failures",
            "win_rate", "score_rate", "win_wilson_lower", "failure_rate",
            "worst_opponent", "worst_score_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in report["learners"])
    print(json.dumps({
        "games": report["games"],
        "learners": len(report["learners"]),
        "incomplete_pairs": len(report["incomplete_pairs"]),
        "ranking": str(args.output_dir / "ranking.csv"),
    }))


if __name__ == "__main__":
    main()
