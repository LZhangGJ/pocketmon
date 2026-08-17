from __future__ import annotations

import collections
import json
from pathlib import Path


ROOT = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")
league = json.loads((ROOT / "state/league.json").read_text())
print("ACTIVE_CHAINS")
for name, chain in league["chains"].items():
    current = chain["current"]
    print(
        name,
        "g", current.get("generation"),
        "package", bool(current.get("packageManifest")),
        "checkpoint", current.get("checkpoint"),
        "deckPool", chain.get("learnerDeckPool", "fixed"),
    )

pool_path = Path(league["poolPath"])
pool = json.loads(pool_path.read_text())
agents = pool.get("agents", pool.get("opponents", []))
print("FROZEN_POOL", pool_path, "COUNT", len(agents))
groups: dict[str, list[str]] = collections.defaultdict(list)
for row in agents:
    name = str(row.get("name") or row.get("agentName") or row.get("id"))
    if name.startswith("universal_bc_0812_standard"):
        group = "new_bc_standard_1m"
    elif name.startswith("universal_bc_0812_large"):
        group = "new_bc_large_256x6"
    elif "hard_exploiter" in name:
        group = "hard_exploiter_g10"
    elif "diversity" in name:
        group = "diversity_g20"
    elif "a08" in name.lower() and "maxbelt" in name.lower():
        group = "a08_maxbelt_anchor"
    elif "a08" in name.lower() and ("277" in name or "original" in name.lower()):
        group = "a08_g277_anchor"
    elif name.startswith("live_") or "ppo" in name.lower():
        group = "historical_ppo"
    else:
        group = "static_frozen"
    groups[group].append(name)
for group, names in sorted(groups.items()):
    print("GROUP", group, len(names))
    for name in names:
        print(" -", name)

print("READY_COUNTS")
for name in league["chains"]:
    paths = list((ROOT / "buffer/ready" / name).glob("*.summary.json"))
    print(name, len(paths), max((p.stat().st_mtime for p in paths), default=0))
