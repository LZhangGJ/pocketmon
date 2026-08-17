import glob
import json
import os
import sys
from collections import defaultdict

root = sys.argv[1]
chain = "a08_dipplin_seaking"
rows = []
for path in sorted(glob.glob(os.path.join(root, "*", "report.json"))):
    try:
        report = json.load(open(path, encoding="utf-8"))
        value = report.get("chains", {}).get(chain)
        if not value:
            continue
        frozen = value.get("frozenAggregate", {})
        direct = value.get("directVsUniversalBc", {})
        seats = value.get("seatMetrics", {})
        agents = {row["agent"]: row for row in value.get("agents", [])}
        def rate(agent):
            return agents.get(agent, {}).get("ppo", {}).get("scoreRate")
        rows.append({
            "round": os.path.basename(os.path.dirname(path)),
            "updatedAt": report.get("updatedAt"),
            "generation": value.get("generation"),
            "games": frozen.get("completed"),
            "wins": frozen.get("wins"),
            "scoreRate": frozen.get("scoreRate"),
            "seat0": seats.get("0", {}).get("scoreRate"),
            "seat1": seats.get("1", {}).get("scoreRate"),
            "directBc": direct.get("scoreRate"),
            "archaludon": rate("public_archaludon_meta"),
            "hardA06": rate("hard_exploiter_g0010__05_a06_89e6155f2531"),
            "diversityA01": rate("diversity_g0020__01_a01_ba51a134262b"),
        })
    except Exception as exc:
        print(json.dumps({"warning": path, "error": str(exc)}), file=sys.stderr)

groups = defaultdict(list)
for row in rows:
    groups[row["generation"]].append(row)
aggregates = []
for generation, values in sorted(groups.items()):
    complete = [v for v in values if v["scoreRate"] is not None]
    if not complete:
        continue
    def mean(key):
        vals = [v[key] for v in complete if v[key] is not None]
        return sum(vals) / len(vals) if vals else None
    aggregates.append({
        "generation": generation,
        "rounds": len(complete),
        "games": sum(v["games"] or 0 for v in complete),
        "scoreRateMean": mean("scoreRate"),
        "scoreRateMin": min(v["scoreRate"] for v in complete),
        "scoreRateMax": max(v["scoreRate"] for v in complete),
        "seat0Mean": mean("seat0"),
        "seat1Mean": mean("seat1"),
        "directBcMean": mean("directBc"),
        "archaludonMean": mean("archaludon"),
        "hardA06Mean": mean("hardA06"),
        "diversityA01Mean": mean("diversityA01"),
        "firstRound": complete[0]["round"],
        "lastRound": complete[-1]["round"],
    })
print(json.dumps({"rounds": rows, "byGeneration": aggregates}, ensure_ascii=False))
