from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/capacity-comparison/screening")
CHAIN = "a08_dipplin_seaking"


for profile in ("standard-frozen40", "large-frozen40"):
    payload = json.loads((ROOT / profile / "monitoring/full-matrix/latest.json").read_text())
    agents = payload["chains"][CHAIN]["agents"]
    if isinstance(agents, dict):
        agents = [dict(value, agent=key) for key, value in agents.items()]
    rows = []
    for cell in agents:
        result = cell.get("universalBc", {})
        rows.append(
            {
                "name": cell.get("agent") or cell.get("name") or cell.get("opponent"),
                "games": result.get("games"),
                "wins": result.get("wins"),
                "draws": result.get("draws"),
                "scoreRate": result.get("scoreRate"),
            }
        )
    print(json.dumps({"profile": profile, "agents": rows}, ensure_ascii=False))
