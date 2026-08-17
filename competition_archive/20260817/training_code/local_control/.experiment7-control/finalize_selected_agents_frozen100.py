#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FAILURES = {"crash", "timeout", "illegal"}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metric(rows: list[dict[str, str]]) -> dict[str, Any]:
    wins = sum(row.get("result") == "win" for row in rows)
    losses = sum(row.get("result") == "loss" for row in rows)
    draws = sum(row.get("result") == "draw" for row in rows)
    failures = sum(row.get("result") in FAILURES for row in rows)
    completed = wins + losses + draws
    return {
        "games": len(rows),
        "completed": completed,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "failures": failures,
        "scoreRate": (wins + 0.5 * draws) / completed if completed else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    args = parser.parse_args()
    staging = args.staging.resolve()
    eval_root = args.eval_root.resolve()
    metadata = load(staging / "metadata.json")
    candidates = load(staging / "learners.json")["agents"]
    opponents = load(staging / "opponents.json")["agents"]

    expected = len(candidates) * len(opponents) * int(metadata["gamesPerOpponent"])
    rows: list[dict[str, str]] = []
    shard_paths = sorted((staging / "raw").glob("results-shard-*.csv"))
    if len(shard_paths) != int(metadata["shards"]):
        raise RuntimeError(f"shard coverage mismatch: {len(shard_paths)} != {metadata['shards']}")
    for path in shard_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if len(rows) != expected:
        raise RuntimeError(f"game coverage mismatch: {len(rows)} != {expected}")

    results = []
    for candidate in candidates:
        candidate_rows = [row for row in rows if row["learner"] == candidate["name"]]
        agents = []
        for opponent in opponents:
            selected = [row for row in candidate_rows if row["opponent"] == opponent["name"]]
            agents.append({"agent": opponent["name"], **metric(selected)})
        results.append(
            {
                **candidate,
                "aggregate": metric(candidate_rows),
                "seat0": metric([row for row in candidate_rows if row["learner_seat"] == "0"]),
                "seat1": metric([row for row in candidate_rows if row["learner_seat"] == "1"]),
                "agents": agents,
            }
        )
    report = {
        **metadata,
        "status": "complete",
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "games": len(rows),
        "results": results,
        "resumedShards": [11, 23],
        "originalFailedHost": "10.113.13.78",
        "originalFailure": "bwrap loopback RTM_NEWADDR not permitted",
    }
    write(staging / "report.json", report)
    final = staging.parent / metadata["roundId"]
    if final.exists():
        raise FileExistsError(final)
    os.replace(staging, final)
    report["roundPath"] = str(final)
    temporary = eval_root / f".latest.{os.getpid()}.tmp"
    write(temporary, report)
    os.replace(temporary, eval_root / "latest.json")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
