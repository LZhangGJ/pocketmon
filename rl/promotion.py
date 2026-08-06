from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any


FAILURES = {"crash", "timeout", "illegal"}


def wilson_lower(wins: int, games: int, z: float = 1.959963984540054) -> float:
    if games <= 0:
        return 0.0
    proportion = wins / games
    denominator = 1.0 + z * z / games
    centre = proportion + z * z / (2.0 * games)
    margin = z * math.sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * games)) / games)
    return (centre - margin) / denominator


def build_promotion_schedule(
    *,
    candidate: str,
    parent: str,
    public_opponents: list[str],
    games_per_public: int,
    parent_games: int,
    seed: int,
) -> list[dict[str, Any]]:
    if games_per_public < 2 or games_per_public % 2 or parent_games < 2 or parent_games % 2:
        raise ValueError("promotion game counts must be positive and even")
    if not public_opponents or len(public_opponents) != len(set(public_opponents)):
        raise ValueError("public opponent names must be non-empty and unique")
    rows: list[dict[str, Any]] = []
    for opponent_index, opponent in enumerate(public_opponents):
        base = seed + opponent_index * 10_000
        for pair in range(games_per_public // 2):
            game_seed = base + pair
            for learner in (candidate, parent):
                rows.append({"learner": learner, "opponent": opponent, "seed": game_seed, "learner_seat": 0})
                rows.append({"learner": learner, "opponent": opponent, "seed": game_seed, "learner_seat": 1})
    head_base = seed + len(public_opponents) * 10_000
    for pair in range(parent_games // 2):
        game_seed = head_base + pair
        rows.append({"learner": candidate, "opponent": parent, "seed": game_seed, "learner_seat": 0})
        rows.append({"learner": candidate, "opponent": parent, "seed": game_seed, "learner_seat": 1})
    random.Random(seed).shuffle(rows)
    return rows


def build_common_opponent_schedule(
    *,
    learners: list[str],
    opponents: list[str],
    games_per_opponent: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Build a paired-seat screen where every learner sees identical games."""

    if games_per_opponent < 2 or games_per_opponent % 2:
        raise ValueError("screen game count must be positive and even")
    if not learners or len(learners) != len(set(learners)):
        raise ValueError("screen learner names must be non-empty and unique")
    if not opponents or len(opponents) != len(set(opponents)):
        raise ValueError("screen opponent names must be non-empty and unique")
    rows: list[dict[str, Any]] = []
    for opponent_index, opponent in enumerate(opponents):
        base = seed + opponent_index * 10_000
        for pair in range(games_per_opponent // 2):
            game_seed = base + pair
            for learner in learners:
                rows.append({"learner": learner, "opponent": opponent, "seed": game_seed, "learner_seat": 0})
                rows.append({"learner": learner, "opponent": opponent, "seed": game_seed, "learner_seat": 1})
    random.Random(seed).shuffle(rows)
    return rows


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(rows)
    wins = sum(row["result"] == "win" for row in rows)
    draws = sum(row["result"] == "draw" for row in rows)
    failures = sum(row["result"] in FAILURES for row in rows)
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "failures": failures,
        "score_rate": (wins + 0.5 * draws) / games if games else 0.0,
        "wilson_lower": wilson_lower(wins, games),
    }


def evaluate_common_opponent_screen(
    rows: list[dict[str, Any]],
    *,
    learners: list[str],
    opponents: list[str],
) -> list[dict[str, Any]]:
    """Rank policy/deck variants evaluated on identical opponent seeds/seats."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["learner"]), str(row["opponent"]))].append(row)
    expected_keys: dict[str, set[tuple[int, int]]] = {}
    ranking = []
    for learner in learners:
        learner_rows = []
        matchup_metrics = {}
        for opponent in opponents:
            matchup = grouped[(learner, opponent)]
            if not matchup:
                raise ValueError(f"common-opponent screen missing {(learner, opponent)}")
            keys = {(int(row["seed"]), int(row["learner_seat"])) for row in matchup}
            if opponent not in expected_keys:
                expected_keys[opponent] = keys
            elif keys != expected_keys[opponent]:
                raise ValueError(f"common-opponent screen did not use identical seeds/seats for {opponent}")
            learner_rows.extend(matchup)
            matchup_metrics[opponent] = _metrics(matchup)
        aggregate = _metrics(learner_rows)
        worst_opponent = min(
            opponents,
            key=lambda opponent: (matchup_metrics[opponent]["score_rate"], opponent),
        )
        ranking.append({
            "learner": learner,
            **aggregate,
            "worst_opponent": worst_opponent,
            "worst_score_rate": matchup_metrics[worst_opponent]["score_rate"],
            "matchups": matchup_metrics,
        })
    ranking.sort(
        key=lambda row: (row["failures"] == 0, row["score_rate"], row["worst_score_rate"], row["wilson_lower"]),
        reverse=True,
    )
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
    return ranking


def evaluate_promotion(
    rows: list[dict[str, Any]],
    *,
    candidate: str,
    parent: str,
    public_opponents: list[str],
    min_head_to_head_score: float = 0.55,
    min_head_to_head_wilson: float = 0.30,
    min_public_delta: float = 0.0,
    max_worst_matchup_regression: float = 0.25,
    max_seat_gap: float = 0.20,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["learner"]), str(row["opponent"]))].append(row)
    missing = [
        (learner, opponent)
        for learner in (candidate, parent)
        for opponent in public_opponents
        if not grouped[(learner, opponent)]
    ]
    if not grouped[(candidate, parent)]:
        missing.append((candidate, parent))
    if missing:
        raise ValueError(f"promotion results are incomplete: {missing}")

    candidate_public_rows = [row for opponent in public_opponents for row in grouped[(candidate, opponent)]]
    parent_public_rows = [row for opponent in public_opponents for row in grouped[(parent, opponent)]]
    candidate_public = _metrics(candidate_public_rows)
    parent_public = _metrics(parent_public_rows)
    head_to_head = _metrics(grouped[(candidate, parent)])
    matchup_deltas = {
        opponent: _metrics(grouped[(candidate, opponent)])["score_rate"]
        - _metrics(grouped[(parent, opponent)])["score_rate"]
        for opponent in public_opponents
    }
    worst_opponent = min(matchup_deltas, key=lambda opponent: (matchup_deltas[opponent], opponent))
    seat_metrics = {
        str(seat): _metrics([row for row in candidate_public_rows if int(row["learner_seat"]) == seat])
        for seat in (0, 1)
    }
    seat_gap = abs(seat_metrics["0"]["score_rate"] - seat_metrics["1"]["score_rate"])
    public_delta = candidate_public["score_rate"] - parent_public["score_rate"]
    failures = candidate_public["failures"] + head_to_head["failures"]
    checks = {
        "zero_failures": failures == 0,
        "head_to_head_score": head_to_head["score_rate"] >= min_head_to_head_score,
        "head_to_head_wilson": head_to_head["wilson_lower"] >= min_head_to_head_wilson,
        "public_delta": public_delta >= min_public_delta,
        "worst_matchup_regression": matchup_deltas[worst_opponent] >= -max_worst_matchup_regression,
        "seat_gap": seat_gap <= max_seat_gap,
    }
    return {
        "promote": all(checks.values()),
        "checks": checks,
        "candidate": candidate,
        "parent": parent,
        "candidate_public": candidate_public,
        "parent_public": parent_public,
        "public_delta": public_delta,
        "head_to_head": head_to_head,
        "matchup_deltas": matchup_deltas,
        "worst_opponent": worst_opponent,
        "worst_matchup_delta": matchup_deltas[worst_opponent],
        "seat_metrics": seat_metrics,
        "seat_gap": seat_gap,
        "thresholds": {
            "min_head_to_head_score": min_head_to_head_score,
            "min_head_to_head_wilson": min_head_to_head_wilson,
            "min_public_delta": min_public_delta,
            "max_worst_matchup_regression": max_worst_matchup_regression,
            "max_seat_gap": max_seat_gap,
        },
    }
