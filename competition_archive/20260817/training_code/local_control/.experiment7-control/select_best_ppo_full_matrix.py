#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


CHAINS = (
    "a02_submission4_grimmsnarl_froslass_munkidori",
    "a05_raging_bolt_ogerpon_kangaskhan",
    "a08_dipplin_seaking",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-root", type=Path, required=True)
    args = parser.parse_args()
    rounds = args.league_root / "monitoring/full-matrix/rounds"
    result = {chain: [] for chain in CHAINS}
    for report_path in sorted(rounds.glob("*/report.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("status") not in (None, "complete"):
            continue
        for chain in CHAINS:
            row = report.get("chains", {}).get(chain)
            if not row:
                continue
            base_agents = [a for a in row.get("agents", []) if not a["agent"].startswith("champion_")]
            completed = sum(int(a["ppo"].get("completed", 0)) for a in base_agents)
            points = sum(float(a["ppo"].get("wins", 0)) + 0.5 * float(a["ppo"].get("draws", 0)) for a in base_agents)
            result[chain].append(
                {
                    "roundId": report.get("roundId") or report_path.parent.name,
                    "generation": int(row["generation"]),
                    "commonFrozenAgents": len(base_agents),
                    "commonGames": completed,
                    "commonScoreRate": points / completed if completed else None,
                    "fullScoreRate": row.get("frozenAggregate", {}).get("scoreRate"),
                    "directBcScoreRate": row.get("directVsUniversalBc", {}).get("scoreRate"),
                    "agentDir": next(
                        (
                            item.get("agentDir")
                            for item in report.get("selected", [])
                            if item.get("chain") == chain
                        ),
                        None,
                    ),
                }
            )
    for chain in CHAINS:
        result[chain].sort(
            key=lambda row: (
                row["commonScoreRate"] if row["commonScoreRate"] is not None else -1.0,
                row["commonGames"],
                row["generation"],
            ),
            reverse=True,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
