#!/usr/bin/env python3
import json
import pathlib

ROOT = pathlib.Path(
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811"
)
TARGETS = ["a02_grim_g247", "a02_grim_g247_pokegear", "a08_maxbelt"]
report = json.loads((ROOT / "monitoring/full-matrix/latest.json").read_text())
out = {
    "roundId": report["roundId"],
    "updatedAt": report["updatedAt"],
    "games": report["games"],
    "engineSeedControlled": report["engineSeedControlled"],
    "chains": {},
}
for name in TARGETS:
    chain = report["chains"][name]
    agents = sorted(
        chain.get("agents", []),
        key=lambda row: (row["ppo"]["scoreRate"], row["agent"]),
    )
    out["chains"][name] = {
        "generation": chain["generation"],
        "frozenAggregate": chain["frozenAggregate"],
        "deltaVsPrevious": chain["deltaVsPrevious"],
        "progress": chain["progress"],
        "seatMetrics": chain["seatMetrics"],
        "seatGap": chain["seatGap"],
        "directVsUniversalBc": chain["directVsUniversalBc"],
        "headToHeadTargets": {
            other: value
            for other, value in chain.get("ppoHeadToHead", {}).items()
            if other in TARGETS
        },
        "weakestFive": [
            {
                "agent": row["agent"],
                "archetype": row["archetype"],
                "games": row["ppo"]["games"],
                "scoreRate": row["ppo"]["scoreRate"],
            }
            for row in agents[:5]
        ],
    }
print(json.dumps(out, indent=2))
