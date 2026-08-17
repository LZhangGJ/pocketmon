import json
import sys
from collections import defaultdict


path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)

groups = ("hard_exploiter_g0010", "diversity_g0020")
totals = defaultdict(lambda: [0, 0, 0])

print(json.dumps({
    "updatedAt": data.get("updatedAt"),
    "roundId": data.get("roundId"),
    "games": data.get("games"),
    "frozenAgentCount": data.get("frozenAgentCount"),
    "chains": list(data.get("chains", {})),
}, ensure_ascii=False))

for chain_name, chain in data.get("chains", {}).items():
    rows = []
    for row in chain.get("agents", []):
        agent = str(row.get("agent", ""))
        group = next((item for item in groups if agent.startswith(item)), None)
        if group is None:
            continue
        metric = row.get("ppo", {})
        games = int(metric.get("completed", metric.get("games", 0)) or 0)
        wins = int(metric.get("wins", 0) or 0)
        losses = int(metric.get("losses", 0) or 0)
        totals[(chain_name, group)][0] += wins
        totals[(chain_name, group)][1] += losses
        totals[(chain_name, group)][2] += games
        totals[("ALL", group)][0] += wins
        totals[("ALL", group)][1] += losses
        totals[("ALL", group)][2] += games
        rows.append((agent, wins, losses, metric.get("scoreRate")))
    print("CHAIN", chain_name)
    for agent, wins, losses, rate in rows:
        print("AGENT", agent, wins, losses, rate)
    for group in groups:
        wins, losses, games = totals[(chain_name, group)]
        print("GROUP", group, wins, losses, games, (wins / games if games else None))

print("ALL")
for group in groups:
    wins, losses, games = totals[("ALL", group)]
    print("GROUP", group, wins, losses, games, (wins / games if games else None))
