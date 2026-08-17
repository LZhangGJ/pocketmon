#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def bc():
    root = Path("/tmp/lzhang-bc-capacity-a100-20260813")
    out = {}
    for profile in ("standard_1m", "large_256x6"):
        report = read(root / f"{profile}-persistent-continuation/training_report.json") or {}
        epochs = report.get("epochs", [])
        out[profile] = [
            {
                "epoch": e.get("epoch"),
                "trainSeconds": (e.get("training") or {}).get("seconds"),
                "validationSeconds": (e.get("validation") or {}).get("seconds"),
                "semantic": (e.get("validation") or {}).get("exactSemantic"),
                "brier": (e.get("validation") or {}).get("valueBrier"),
                "earlyStopping": e.get("earlyStopping"),
            }
            for e in epochs
        ]
    print(json.dumps(out, indent=2))


def a02():
    root = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")
    current = read(root / "monitoring/full-matrix/latest.json") or {}
    rows = []
    for path in sorted((root / "monitoring/full-matrix").glob("**/*.json")):
        payload = read(path)
        if not isinstance(payload, dict) or payload.get("status") != "complete" or payload.get("busy"):
            continue
        row = (payload.get("chains") or {}).get("a02_submission4_grimmsnarl_froslass_munkidori")
        if not isinstance(row, dict):
            continue
        agg = row.get("frozenAggregate") or {}
        direct = row.get("directVsUniversalBc") or {}
        seats = row.get("seatMetrics") or {}
        rows.append({
            "path": str(path), "roundId": payload.get("roundId"), "updatedAt": payload.get("updatedAt"),
            "generation": row.get("generation"), "games": agg.get("games"), "scoreRate": agg.get("scoreRate"),
            "delta": row.get("deltaVsPrevious"), "progress": row.get("progress"),
            "directBcGames": direct.get("games"), "directBc": direct.get("scoreRate"),
            "seat0": (seats.get("0") or {}).get("scoreRate"), "seat1": (seats.get("1") or {}).get("scoreRate"),
        })
    # De-duplicate multiple aliases (latest/current/history) by round id.
    unique = {}
    for row in rows:
        unique[row["roundId"] or row["path"]] = row
    ordered = sorted(unique.values(), key=lambda r: (r.get("updatedAt") or "", r.get("generation") or -1))
    league = read(root / "state/league.json") or {}
    chain = (league.get("chains") or {}).get("a02_submission4_grimmsnarl_froslass_munkidori", {})
    print(json.dumps({"currentGeneration": (chain.get("current") or {}).get("generation"), "matrixHistory": ordered[-12:]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("bc", "a02"))
    args = parser.parse_args()
    bc() if args.mode == "bc" else a02()
