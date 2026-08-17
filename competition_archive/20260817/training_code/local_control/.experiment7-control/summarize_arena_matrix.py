from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_FIELDS = ("learner", "opponent", "seed", "learner_seat")
FAILURE_RESULTS = {"crash", "timeout", "illegal"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(row.get(field, "") for field in KEY_FIELDS)  # type: ignore[return-value]


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row_key = key(row)
                if row_key in seen:
                    raise ValueError(f"duplicate Arena result key: {row_key}")
                seen.add(row_key)
                rows.append(row)
    return rows


def wilson(score: float, games: int, z: float = 1.959963984540054) -> list[float]:
    if games <= 0:
        return [0.0, 0.0]
    denominator = 1.0 + z * z / games
    centre = score + z * z / (2.0 * games)
    margin = z * math.sqrt((score * (1.0 - score) + z * z / (4.0 * games)) / games)
    return [(centre - margin) / denominator, (centre + margin) / denominator]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    wins = sum(row.get("result") == "win" for row in rows)
    losses = sum(row.get("result") == "loss" for row in rows)
    draws = sum(row.get("result") == "draw" for row in rows)
    failures = sum(row.get("result") in FAILURE_RESULTS for row in rows)
    completed = wins + losses + draws
    score = (wins + 0.5 * draws) / completed if completed else 0.0
    latencies = [float(row["latency_ms"]) for row in rows if row.get("latency_ms")]
    return {
        "games": len(rows),
        "completed": completed,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "failures": failures,
        "scoreRate": score,
        "wilson95": wilson(score, completed),
        "failureRate": failures / len(rows) if rows else 0.0,
        "latencyMsP50": statistics.median(latencies) if latencies else 0.0,
        "latencyMsP95": percentile(latencies, 0.95),
        "latencyMsMax": max(latencies, default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize a paired-seat Arena payoff matrix")
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--learners", type=Path, required=True)
    parser.add_argument("--opponents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    schedule_path = args.schedule.resolve()
    result_paths = [path.resolve() for path in args.results]
    learners_path = args.learners.resolve()
    opponents_path = args.opponents.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite matrix summary: {output_dir}")
    schedule_rows = read_rows([schedule_path])
    result_rows = read_rows(result_paths)
    schedule_keys = {key(row) for row in schedule_rows}
    result_keys = {key(row) for row in result_rows}
    if schedule_keys != result_keys:
        raise ValueError(
            f"result coverage mismatch: schedule={len(schedule_keys)} results={len(result_keys)} "
            f"missing={len(schedule_keys - result_keys)} extra={len(result_keys - schedule_keys)}"
        )

    learner_payload = json.loads(learners_path.read_text(encoding="utf-8"))
    learner_rows = learner_payload.get("agents", learner_payload.get("packages", []))
    opponent_payload = json.loads(opponents_path.read_text(encoding="utf-8"))
    opponent_rows = opponent_payload.get("agents", [])
    learner_names = [row["name"] for row in learner_rows if row.get("status", "accepted") == "accepted"]
    opponent_meta = {
        row["name"]: {
            "archetype": row.get("archetype", "unknown"),
            "skillTier": row.get("skill_tier", "unknown"),
            "directorySha256": row.get("directory_sha256", row.get("directorySha256", "")),
        }
        for row in opponent_rows
        if row.get("status", "accepted") == "accepted"
    }
    if set(learner_names) != {row["learner"] for row in result_rows}:
        raise ValueError("learner manifest and results do not contain the same names")
    if set(opponent_meta) != {row["opponent"] for row in result_rows}:
        raise ValueError("opponent manifest and results do not contain the same names")

    by_matchup: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_learner: dict[str, list[dict[str, str]]] = defaultdict(list)
    paired: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in result_rows:
        by_matchup[(row["learner"], row["opponent"])].append(row)
        by_learner[row["learner"]].append(row)
        paired[(row["learner"], row["opponent"], row["seed"])].add(row["learner_seat"])
    incomplete_pairs = [(*pair, sorted(seats)) for pair, seats in paired.items() if seats != {"0", "1"}]
    if incomplete_pairs:
        raise ValueError(f"paired-seat groups are incomplete: {incomplete_pairs[:3]}")

    matchup_records = []
    learners = []
    for learner in learner_names:
        matchup_metrics: dict[str, dict[str, Any]] = {}
        archetype_scores: dict[str, list[float]] = defaultdict(list)
        hard_scores = []
        for opponent in sorted(opponent_meta):
            record = metrics(by_matchup[(learner, opponent)])
            meta = opponent_meta[opponent]
            matchup_metrics[opponent] = record
            archetype_scores[meta["archetype"]].append(record["scoreRate"])
            if meta["skillTier"] == "hard":
                hard_scores.append(record["scoreRate"])
            matchup_records.append(
                {
                    "learner": learner,
                    "opponent": opponent,
                    "archetype": meta["archetype"],
                    "skillTier": meta["skillTier"],
                    **record,
                }
            )
        aggregate = metrics(by_learner[learner])
        agent_scores = [row["scoreRate"] for row in matchup_metrics.values()]
        archetype_means = {
            archetype: statistics.mean(scores) for archetype, scores in archetype_scores.items()
        }
        worst_opponent = min(matchup_metrics, key=lambda name: (matchup_metrics[name]["scoreRate"], name))
        seat_metrics = {
            seat: metrics([row for row in by_learner[learner] if row["learner_seat"] == seat])
            for seat in ("0", "1")
        }
        submission4 = matchup_metrics.get("team_submission_4_portable_bc")
        learners.append(
            {
                "learner": learner,
                **aggregate,
                "agentMacroScoreRate": statistics.mean(agent_scores),
                "archetypeMacroScoreRate": statistics.mean(archetype_means.values()),
                "hardTierAgentMacroScoreRate": statistics.mean(hard_scores) if hard_scores else None,
                "worstOpponent": worst_opponent,
                "worstOpponentScoreRate": matchup_metrics[worst_opponent]["scoreRate"],
                "seatMetrics": seat_metrics,
                "seatGap": abs(seat_metrics["0"]["scoreRate"] - seat_metrics["1"]["scoreRate"]),
                "submission4ScoreRate": submission4["scoreRate"] if submission4 else None,
                "archetypes": archetype_means,
            }
        )
    learners.sort(
        key=lambda row: (
            row["failures"] == 0,
            row["hardTierAgentMacroScoreRate"] or -1.0,
            row["agentMacroScoreRate"],
            row["worstOpponentScoreRate"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(learners, start=1):
        row["rank"] = rank

    created_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "schemaVersion": 1,
        "createdAt": created_at,
        "schedule": {"path": str(schedule_path), "sha256": sha256_file(schedule_path), "games": len(schedule_rows)},
        "sources": {
            "results": [{"path": str(path), "sha256": sha256_file(path)} for path in result_paths],
            "learners": {"path": str(learners_path), "sha256": sha256_file(learners_path)},
            "opponents": {"path": str(opponents_path), "sha256": sha256_file(opponents_path)},
        },
        "games": len(result_rows),
        "learners": learners,
        "matchups": matchup_records,
        "pairedSeatGroups": len(paired),
        "incompletePairs": [],
        "engineSeedControlled": all(
            row.get("engine_seed_controlled", "").lower() == "true" for row in result_rows
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    ranking_fields = [
        "rank", "learner", "games", "wins", "losses", "draws", "failures", "scoreRate",
        "agentMacroScoreRate", "archetypeMacroScoreRate", "hardTierAgentMacroScoreRate",
        "worstOpponent", "worstOpponentScoreRate", "seatGap", "submission4ScoreRate",
        "latencyMsP50", "latencyMsP95",
    ]
    with (output_dir / "ranking.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ranking_fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in ranking_fields} for row in learners)
    matchup_fields = [
        "learner", "opponent", "archetype", "skillTier", "games", "wins", "losses", "draws",
        "failures", "scoreRate", "failureRate", "latencyMsP50", "latencyMsP95", "latencyMsMax",
    ]
    with (output_dir / "payoff_matrix.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=matchup_fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in matchup_fields} for row in matchup_records)
    print(json.dumps({
        "games": len(result_rows),
        "learners": len(learners),
        "opponents": len(opponent_meta),
        "matchups": len(matchup_records),
        "incompletePairs": 0,
        "ranking": str(output_dir / "ranking.csv"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
