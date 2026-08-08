from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import write_json


def wilson(successes: float, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = p + z * z / (2.0 * trials)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials)
    return (centre - margin) / denominator, (centre + margin) / denominator


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize challenger Arena CSV and apply the frozen promotion gate")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-games", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--min-wilson-low", type=float, default=0.50)
    parser.add_argument("--min-seat-score", type=float, default=0.45)
    args = parser.parse_args()

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.results.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["learner"]].append(row)
    output: dict[str, Any] = {"schema_version": 1, "challengers": {}}
    for learner, rows in sorted(groups.items()):
        counts = Counter(row["result"] for row in rows)
        failures = sum(counts[key] for key in ("crash", "timeout", "illegal"))
        wins, draws, losses = counts["win"], counts["draw"], counts["loss"]
        completed = wins + draws + losses
        points = wins + 0.5 * draws
        score = points / completed if completed else 0.0
        low, high = wilson(points, completed)
        seats = {}
        for seat in (0, 1):
            subset = [row for row in rows if int(row["learner_seat"]) == seat]
            seat_counts = Counter(row["result"] for row in subset)
            seat_completed = seat_counts["win"] + seat_counts["draw"] + seat_counts["loss"]
            seat_points = seat_counts["win"] + 0.5 * seat_counts["draw"]
            seats[str(seat)] = {
                "games": len(subset),
                "score_rate": seat_points / seat_completed if seat_completed else 0.0,
            }
        gate = (
            len(rows) >= args.min_games
            and failures == 0
            and completed == len(rows)
            and score >= args.min_score
            and low > args.min_wilson_low
            and min(value["score_rate"] for value in seats.values()) >= args.min_seat_score
        )
        output["challengers"][learner] = {
            "games": len(rows),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score_rate": score,
            "wilson_low": low,
            "wilson_high": high,
            "failures": failures,
            "failure_counts": {key: counts[key] for key in ("crash", "timeout", "illegal")},
            "seats": seats,
            "gate_passed": gate,
        }
    write_json(args.output, output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
