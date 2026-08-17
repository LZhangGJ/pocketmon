import json

path = "/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/monitoring/full-matrix/latest.json"
data = json.load(open(path, encoding="utf-8"))
out = {}
for name, chain in data["chains"].items():
    agents = sorted(
        [
            {
                "agent": row["agent"],
                "ppo": row["ppo"]["scoreRate"],
                "bc": row["universalBc"]["scoreRate"],
                "delta": row["deltaVsPrevious"],
            }
            for row in chain["agents"]
        ],
        key=lambda row: (row["ppo"], row["delta"]),
    )[:6]
    out[name] = {
        "weak": agents,
        "h2h": {
            opponent: round(result["scoreRate"], 4)
            for opponent, result in chain["ppoHeadToHead"].items()
        },
    }
print(json.dumps(out, separators=(",", ":")))
