from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/"
    "capacity-comparison/screening"
)
RUNS = ("standard-smoke", "standard-frozen40", "large-smoke", "large-frozen40")
KEY_AGENTS = ("public_archaludon_meta", "hard_g10_a06", "diversity_g20_a01")


def aggregate(cells: list[dict[str, Any]]) -> dict[str, float | int | None]:
    games = sum(int(row.get("games", 0)) for row in cells)
    wins = sum(int(row.get("wins", 0)) for row in cells)
    draws = sum(int(row.get("draws", 0)) for row in cells)
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "scoreRate": (wins + 0.5 * draws) / games if games else None,
    }


def main() -> None:
    output: dict[str, Any] = {}
    for run in RUNS:
        path = ROOT / run / "monitoring/full-matrix/latest.json"
        if not path.is_file():
            output[run] = {"exists": False}
            continue
        payload = json.loads(path.read_text())
        chains_out: dict[str, Any] = {}
        candidate_cells: list[dict[str, Any]] = []
        for chain, row in payload.get("chains", {}).items():
            frozen = row.get("agents", [])
            if isinstance(frozen, dict):
                frozen = [dict(value, agent=key) for key, value in frozen.items()]
            candidate = [cell.get("universalBc", {}) for cell in frozen]
            candidate_cells.extend(candidate)
            keys = {}
            for wanted in KEY_AGENTS:
                matches = [
                    cell for cell in frozen if wanted in str(cell.get("agent", "")).lower()
                ]
                keys[wanted] = aggregate([cell.get("universalBc", {}) for cell in matches])
            chains_out[chain] = {
                "generation": row.get("generation"),
                "ppoFrozen": row.get("frozenAggregate"),
                "candidateBcFrozen": row.get("universalBcFrozenAggregate"),
                "directPpoVsCandidateBc": row.get("directVsUniversalBc"),
                "keyAgents": keys,
                "rowKeys": sorted(row),
            }
        output[run] = {
            "exists": True,
            "status": payload.get("status"),
            "createdAt": payload.get("createdAt"),
            "completedAt": payload.get("completedAt"),
            "games": payload.get("games"),
            "engineSeedControlled": payload.get("engineSeedControlled"),
            "candidateBcAllChains": aggregate(candidate_cells),
            "chains": chains_out,
            "topLevelKeys": sorted(payload),
        }
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
