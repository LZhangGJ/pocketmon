from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from common import Experiment7Error, read_json, sha256_file, utc_now, wilson_interval, write_csv, write_json


def make_schedule(
    packages_manifest: Path,
    target_agent: Path,
    output_dir: Path,
    games_per_challenger: int,
    seed_base: int,
    stage: str,
    selected_names: set[str] | None,
) -> dict[str, Any]:
    packages = read_json(packages_manifest)["packages"]
    if selected_names is not None:
        packages = [row for row in packages if row["name"] in selected_names]
    if not packages:
        raise Experiment7Error("arena schedule has no challenger packages")
    if games_per_challenger <= 0 or games_per_challenger % 2:
        raise ValueError("games_per_challenger must be a positive even number")
    output_dir.mkdir(parents=True, exist_ok=True)
    learners = {
        "agents": [
            {"name": row["name"], "agent_dir": row["agentDir"], "status": "accepted"}
            for row in packages
        ]
    }
    opponents = {
        "agents": [
            {"name": "frozen_lucario_rule", "agent_dir": str(target_agent.resolve()), "status": "accepted"}
        ]
    }
    learners_path = output_dir / "learners.json"
    opponents_path = output_dir / "opponents.json"
    write_json(learners_path, learners)
    write_json(opponents_path, opponents)
    rows = []
    per_seat = games_per_challenger // 2
    game_id = 0
    for learner_index, row in enumerate(packages):
        for seat in (0, 1):
            for local_index in range(per_seat):
                rows.append(
                    {
                        "game_id": f"{stage}-{row['name']}-{seat}-{local_index:04d}",
                        "learner": row["name"],
                        "opponent": "frozen_lucario_rule",
                        "seed": seed_base + learner_index * 1_000_000 + seat * 100_000 + local_index,
                        "learner_seat": seat,
                    }
                )
                game_id += 1
    schedule = output_dir / "schedule.csv"
    write_csv(schedule, rows, ["game_id", "learner", "opponent", "seed", "learner_seat"])
    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "stage": stage,
        "gamesPerChallenger": games_per_challenger,
        "seatBalanced": True,
        "challengers": [row["name"] for row in packages],
        "targetAgent": {"path": str(target_agent.resolve()), "directorySha256": directory_sha256(target_agent)},
        "schedule": {"path": str(schedule.resolve()), "sha256": sha256_file(schedule), "games": len(rows)},
        "learners": str(learners_path.resolve()),
        "opponents": str(opponents_path.resolve()),
    }
    write_json(output_dir / "schedule_receipt.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def directory_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = child.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _candidate_stats(row: dict[str, str]) -> dict[str, Any]:
    raw = row.get("diagnostics_json", "")
    try:
        diagnostics = json.loads(raw) if raw else []
    except json.JSONDecodeError:
        return {"diagnosticsParseError": 1}
    if not isinstance(diagnostics, list):
        return {"diagnosticsParseError": 1}
    seat = int(row["learner_seat"])
    if seat >= len(diagnostics) or not isinstance(diagnostics[seat], dict):
        return {}
    value = diagnostics[seat]
    if "bc" in value and isinstance(value["bc"], dict):
        return value["bc"]
    return value


def summarize_results(
    result_paths: list[Path],
    output: Path,
    stage: str,
    minimum_score: float,
    minimum_wilson: float,
    minimum_seat_score: float,
    top_n: int,
) -> dict[str, Any]:
    rows = []
    seen_keys = set()
    for path in result_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row.get("game_id", ""),
                    row.get("learner", ""),
                    row.get("seed", ""),
                    row.get("learner_seat", ""),
                )
                if key in seen_keys:
                    raise Experiment7Error(f"duplicate arena result row: {key}")
                seen_keys.add(key)
                rows.append(row)
    if not rows:
        raise Experiment7Error("no arena result rows")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["learner"]].append(row)
    summaries = []
    for learner, learner_rows in grouped.items():
        result_counts = Counter(row["result"] for row in learner_rows)
        wins = result_counts["win"]
        losses = result_counts["loss"]
        draws = result_counts["draw"]
        completed = wins + losses + draws
        failures = len(learner_rows) - completed
        low, high = wilson_interval(wins, losses, draws)
        score = (wins + 0.5 * draws) / completed if completed else 0.0
        seat_metrics = {}
        for seat in (0, 1):
            seat_rows = [row for row in learner_rows if int(row["learner_seat"]) == seat]
            counts = Counter(row["result"] for row in seat_rows)
            n = counts["win"] + counts["loss"] + counts["draw"]
            seat_metrics[str(seat)] = {
                "games": len(seat_rows),
                "wins": counts["win"],
                "losses": counts["loss"],
                "draws": counts["draw"],
                "failures": len(seat_rows) - n,
                "scoreRate": (counts["win"] + 0.5 * counts["draw"]) / n if n else 0.0,
            }
        diagnostic_sums: Counter[str] = Counter()
        diagnostic_max: Counter[str] = Counter()
        for row in learner_rows:
            stats = _candidate_stats(row)
            for key, value in stats.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    diagnostic_sums[key] += float(value)
                    diagnostic_max[key] = max(diagnostic_max[key], float(value))
        model_calls = diagnostic_max.get("modelCalls", 0.0)
        fallback_calls = diagnostic_max.get("fallbackCalls", 0.0)
        passes_runtime = failures == 0 and model_calls > 0 and fallback_calls == 0
        if stage.lower() == "smoke":
            passes_score = True
        else:
            passes_score = (
                score >= minimum_score
                and low > minimum_wilson
                and all(value["scoreRate"] >= minimum_seat_score for value in seat_metrics.values())
            )
        summaries.append(
            {
                "learner": learner,
                "games": len(learner_rows),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "failures": failures,
                "scoreRate": score,
                "wilson95": [low, high],
                "seats": seat_metrics,
                "diagnosticMax": dict(diagnostic_max),
                "passesRuntimeGate": passes_runtime,
                "passesScoreGate": passes_score,
                "passes": passes_runtime and passes_score,
            }
        )
    summaries.sort(
        key=lambda row: (
            -int(row["passesRuntimeGate"]),
            -float(row["scoreRate"]),
            -float(row["wilson95"][0]),
            row["learner"],
        )
    )
    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "stage": stage,
        "thresholds": {
            "minimumScoreRate": minimum_score,
            "minimumWilsonLower": minimum_wilson,
            "minimumSeatScoreRate": minimum_seat_score,
            "zeroFailures": True,
            "modelCallsPositive": True,
            "fallbackCallsZero": True,
        },
        "sources": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in result_paths],
        "challengers": summaries,
        "topNames": [row["learner"] for row in summaries if row["passesRuntimeGate"]][:top_n],
    }
    write_json(output, payload)
    write_csv(
        output.with_suffix(".csv"),
        [
            {
                "rank": index,
                "learner": row["learner"],
                "games": row["games"],
                "wins": row["wins"],
                "losses": row["losses"],
                "draws": row["draws"],
                "failures": row["failures"],
                "score_rate": row["scoreRate"],
                "wilson_low": row["wilson95"][0],
                "wilson_high": row["wilson95"][1],
                "seat0_score": row["seats"]["0"]["scoreRate"],
                "seat1_score": row["seats"]["1"]["scoreRate"],
                "model_calls": row["diagnosticMax"].get("modelCalls", 0),
                "fallback_calls": row["diagnosticMax"].get("fallbackCalls", 0),
                "passes": row["passes"],
            }
            for index, row in enumerate(summaries, start=1)
        ],
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and summarize Experiment 7 Challenger-vs-Lucario Arena stages")
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make-schedule")
    make.add_argument("--packages", type=Path, required=True)
    make.add_argument("--target-agent", type=Path, required=True)
    make.add_argument("--output-dir", type=Path, required=True)
    make.add_argument("--games-per-challenger", type=int, required=True)
    make.add_argument("--seed-base", type=int, default=2026080800)
    make.add_argument("--stage", required=True)
    make.add_argument("--selected-names", nargs="*")

    summary = sub.add_parser("summarize")
    summary.add_argument("--results", type=Path, nargs="+", required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--stage", required=True)
    summary.add_argument("--minimum-score", type=float, default=0.0)
    summary.add_argument("--minimum-wilson", type=float, default=0.0)
    summary.add_argument("--minimum-seat-score", type=float, default=0.0)
    summary.add_argument("--top-n", type=int, default=3)

    args = parser.parse_args()
    if args.command == "make-schedule":
        make_schedule(
            args.packages.resolve(),
            args.target_agent.resolve(),
            args.output_dir.resolve(),
            args.games_per_challenger,
            args.seed_base,
            args.stage,
            set(args.selected_names) if args.selected_names else None,
        )
    else:
        summarize_results(
            [path.resolve() for path in args.results],
            args.output.resolve(),
            args.stage,
            args.minimum_score,
            args.minimum_wilson,
            args.minimum_seat_score,
            args.top_n,
        )


if __name__ == "__main__":
    main()
