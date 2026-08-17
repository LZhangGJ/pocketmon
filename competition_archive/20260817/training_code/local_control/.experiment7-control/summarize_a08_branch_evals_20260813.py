import json
from pathlib import Path

root = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-deck-branch-evals-20260813")
targets = {
    "arch": "public_archaludon_meta",
    "hardA06": "hard_exploiter_g0010__05_a06_89e6155f2531",
    "divA01": "diversity_g0020__01_a01_ba51a134262b",
}
out = {}
for candidate in sorted(root.glob("a08_*_g*")):
    latest = candidate / "monitoring/full-matrix/latest.json"
    if not latest.is_file():
        continue
    data = json.loads(latest.read_text(encoding="utf-8"))
    chain = data["chains"]["a08_dipplin_seaking"]
    agents = {row["agent"]: row for row in chain["agents"]}
    out[candidate.name] = {
        "status": data["status"],
        "games": data["games"],
        "failures": chain["frozenAggregate"]["failures"],
        "frozen": chain["frozenAggregate"]["scoreRate"],
        "bcFrozen": chain["universalBcFrozenAggregate"]["scoreRate"],
        "seat0": chain["seatMetrics"]["0"]["scoreRate"],
        "seat1": chain["seatMetrics"]["1"]["scoreRate"],
        "gap": chain["seatGap"],
        "directBC": chain["directVsUniversalBc"]["scoreRate"],
        **{label: agents[name]["ppo"]["scoreRate"] for label, name in targets.items()},
    }
print(json.dumps(out, separators=(",", ":")))
