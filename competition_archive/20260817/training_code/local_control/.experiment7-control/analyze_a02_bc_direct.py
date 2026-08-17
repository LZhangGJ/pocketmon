#!/usr/bin/env python3
"""Aggregate A02 direct-vs-Universal-BC Arena results by seat."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path


CHAIN = "a02_submission4_grimmsnarl_froslass_munkidori"


def add_direct_results(round_dir: Path, target: Counter[object]) -> None:
    metadata = json.loads((round_dir / "metadata.json").read_text())
    selected = next(item for item in metadata["selected"] if item["chain"] == CHAIN)
    learner = selected["learner"]
    opponent = selected["bcLearner"]
    for result_path in (round_dir / "raw").glob("*.csv"):
        with result_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["learner"] != learner or row["opponent"] != opponent:
                    continue
                result = row["result"]
                seat = row["learner_seat"]
                target[result] += 1
                target[(seat, result)] += 1


def print_summary(label: str, counts: Counter[object]) -> None:
    results = ("win", "loss", "draw")
    games = sum(counts[result] for result in results)
    score = (counts["win"] + 0.5 * counts["draw"]) / games
    z = 1.96
    denominator = 1 + z * z / games
    midpoint = (score + z * z / (2 * games)) / denominator
    half_width = (
        z
        * math.sqrt(score * (1 - score) / games + z * z / (4 * games * games))
        / denominator
    )
    print(
        label,
        f"games={games}",
        f"W-L-D={counts['win']}-{counts['loss']}-{counts['draw']}",
        f"score={score:.4f}",
        f"wilson95={midpoint - half_width:.4f}..{midpoint + half_width:.4f}",
    )
    for seat in ("0", "1"):
        seat_games = sum(counts[(seat, result)] for result in results)
        seat_score = (
            counts[(seat, "win")] + 0.5 * counts[(seat, "draw")]
        ) / seat_games
        print(
            f"  seat={seat}",
            f"games={seat_games}",
            f"W-L-D={counts[(seat, 'win')]}-{counts[(seat, 'loss')]}-"
            f"{counts[(seat, 'draw')]}",
            f"score={seat_score:.4f}",
        )


def main() -> None:
    base = Path(sys.argv[1])
    latest = json.loads((base / "latest.json").read_text())
    latest_round_id = latest["roundId"]
    latest_generation = latest["chains"][CHAIN]["generation"]
    latest_counts: Counter[object] = Counter()
    generation_counts: Counter[object] = Counter()
    generation_rounds = 0

    for report_path in sorted(base.glob("rounds/*/report.json")):
        report = json.loads(report_path.read_text())
        chain = report.get("chains", {}).get(CHAIN)
        if report.get("status") != "complete" or not chain:
            continue
        if chain.get("generation") != latest_generation:
            continue
        round_counts: Counter[object] = Counter()
        add_direct_results(report_path.parent, round_counts)
        generation_counts.update(round_counts)
        generation_rounds += 1
        if report_path.parent.name == latest_round_id:
            latest_counts.update(round_counts)

    print_summary("latest", latest_counts)
    print_summary(
        f"generation_{latest_generation}_{generation_rounds}_rounds",
        generation_counts,
    )


if __name__ == "__main__":
    main()
