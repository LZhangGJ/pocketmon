import json


PATH = "/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/monitoring/full-matrix/latest.json"
d = json.load(open(PATH, encoding="utf-8"))
out = {"updatedAt": d.get("updatedAt"), "roundId": d.get("roundId"), "games": d.get("games"), "agents": d.get("frozenAgentCount"), "chains": {}, "agentRows": {}}
for name, row in d.get("chains", {}).items():
    frozen = row.get("frozenAggregate", {})
    bc = row.get("universalBcFrozenAggregate", {})
    seats = row.get("seatMetrics", {})
    direct = row.get("directVsUniversalBc", {})
    out["chains"][name] = {
        "generation": row.get("generation"), "ppo": frozen.get("scoreRate"), "bc": bc.get("scoreRate"),
        "ppoMinusBc": row.get("ppoMinusBc"), "delta": row.get("deltaVsPrevious"),
        "seat0": seats.get("0", {}).get("scoreRate"), "seat1": seats.get("1", {}).get("scoreRate"),
        "seatGap": row.get("seatGap"), "directBc": direct.get("scoreRate"),
    }
    for cell in row.get("agents", []):
        a = cell.get("agent")
        out["agentRows"].setdefault(a, {})[name] = {
            "ppo": cell.get("ppo", {}).get("scoreRate"), "bc": cell.get("universalBc", {}).get("scoreRate"),
            "diff": cell.get("ppoMinusBc"), "delta": cell.get("deltaVsPrevious"),
        }
print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
