from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path(
        "/suedata1/Free/lzhang/pocketmon-runs/experiment7-hourly-cache-pool/"
        "monitoring/hourly-cache-pool"
    )
)
FAILURES = {"crash", "timeout", "illegal"}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def score(rows: list[dict[str, str]]) -> dict:
    counts = Counter(row.get("result", "") for row in rows)
    completed = counts["win"] + counts["loss"] + counts["draw"]
    return {
        "games": len(rows),
        "wins": counts["win"],
        "losses": counts["loss"],
        "draws": counts["draw"],
        "failures": sum(counts[name] for name in FAILURES),
        "scoreRate": (
            (counts["win"] + 0.5 * counts["draw"]) / completed if completed else None
        ),
    }


latest_path = ROOT / "latest.json"
rounds = sorted(
    (ROOT / "rounds").glob(".*.in-progress"),
    key=lambda path: path.stat().st_mtime,
)
newer_running = bool(
    rounds
    and (
        not latest_path.is_file()
        or rounds[-1].stat().st_mtime > latest_path.stat().st_mtime
    )
)
if latest_path.is_file() and not newer_running:
    payload = load(latest_path)
    print(json.dumps({"mode": "complete", "report": payload}, ensure_ascii=False))
    raise SystemExit(0)

if not rounds:
    print(json.dumps({"mode": "missing", "root": str(ROOT)}, ensure_ascii=False))
    raise SystemExit(0)

round_path = rounds[-1]
metadata = load(round_path / "metadata.json")
schedule_rows = 0
with (round_path / "schedule.csv").open("r", encoding="utf-8", newline="") as handle:
    schedule_rows = max(sum(1 for _ in handle) - 1, 0)

raw_rows: list[dict[str, str]] = []
for path in sorted((round_path / "raw").glob("results-shard-*.csv")):
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            raw_rows.extend(csv.DictReader(handle))
    except (OSError, csv.Error):
        continue

frozen = set(metadata.get("frozenAgents", []))
chains = {}
for selected in metadata.get("selected", []):
    learner = selected["learner"]
    rows = [
        row
        for row in raw_rows
        if row.get("learner") == learner and row.get("opponent") in frozen
    ]
    seat0 = [row for row in rows if row.get("learner_seat") == "0"]
    seat1 = [row for row in rows if row.get("learner_seat") == "1"]
    chains[selected["chain"]] = {
        "generation": selected["generation"],
        "frozen": score(rows),
        "seat0": score(seat0),
        "seat1": score(seat1),
    }

print(
    json.dumps(
        {
            "mode": "running",
            "roundId": metadata.get("roundId"),
            "startedAt": metadata.get("startedAt"),
            "models": len(metadata.get("selected", [])),
            "scheduleGames": schedule_rows,
            "rawGames": len(raw_rows),
            "progress": len(raw_rows) / schedule_rows if schedule_rows else None,
            "rawShardFiles": len(list((round_path / "raw").glob("results-shard-*.csv"))),
            "logFiles": len(list((round_path / "logs").glob("*.log"))),
            "chains": chains,
        },
        ensure_ascii=False,
    )
)
