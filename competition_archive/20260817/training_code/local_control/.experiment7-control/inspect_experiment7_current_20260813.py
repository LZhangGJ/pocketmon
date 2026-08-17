#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}", "_path": str(path)}


def bc() -> dict:
    root = Path("/tmp/lzhang-bc-capacity-a100-20260813")
    shared = Path(
        "/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/"
        "capacity-comparison-a100-ram-prefetch-b256"
    )
    result = {}
    for profile in ("standard_1m", "large_256x6"):
        local = root / f"{profile}-persistent-continuation"
        state = load(local / "continuation_state.json")
        report = load(local / "training_report.json")
        if "_error" in report:
            report = load(shared / profile / "training_report.json")
        history = report.get("history") or report.get("epochs") or []
        last = history[-1] if history else {}
        best = report.get("best") or report.get("selected") or {}
        result[profile] = {
            "state": state,
            "reportPath": report.get("_path", str(local / "training_report.json")),
            "reportKeys": sorted(report) if isinstance(report, dict) else [],
            "historyCount": len(history),
            "last": last,
            "best": best,
            "selectedEpoch": report.get("selectedEpoch") or report.get("bestEpoch"),
            "earlyStop": report.get("earlyStop") or report.get("earlyStopping"),
        }
    return result


def ppo() -> dict:
    root = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")
    league = load(root / "state/league.json")
    matrix = load(root / "monitoring/full-matrix/latest.json")
    chains = {}
    for name, row in league.get("chains", {}).items():
        current = row.get("current", {})
        chains[name] = {
            "generation": current.get("generation"),
            "snapshotId": current.get("snapshotId"),
            "trainingControl": row.get("trainingControl"),
        }
    matrix_chains = {}
    for name, row in matrix.get("chains", {}).items():
        matrix_chains[name] = {
            "generation": row.get("generation"),
            "progress": row.get("progress"),
            "deltaVsPrevious": row.get("deltaVsPrevious"),
            "frozenAggregate": row.get("frozenAggregate"),
            "universalBcAggregate": row.get("universalBcAggregate"),
            "ppoMinusBc": row.get("ppoMinusBc"),
            "seatMetrics": row.get("seatMetrics"),
            "directVsUniversalBc": row.get("directVsUniversalBc"),
            "ppoHeadToHead": row.get("ppoHeadToHead"),
        }
    return {
        "leagueUpdatedAt": league.get("updatedAt"),
        "chains": chains,
        "matrixMeta": {
            key: matrix.get(key)
            for key in ("status", "busy", "roundId", "updatedAt", "createdAt", "agentCount", "games")
        },
        "matrixChains": matrix_chains,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("bc", "ppo"))
    args = parser.parse_args()
    print(json.dumps(bc() if args.mode == "bc" else ppo(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
