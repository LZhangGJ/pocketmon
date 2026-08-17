#!/usr/bin/env python3
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
targets = [
    "a02_grim_g247",
    "a02_grim_g247_pokegear",
    "a08_rabsca",
    "a08_maxbelt",
    "lucario_gold_exact",
    "universal_ppo_standard_1m",
    "universal_ppo_large_256x6",
]
result = {"leagueUpdatedAt": payload.get("leagueUpdatedAt"), "chains": {}}
for name in targets:
    row = payload["chains"][name]
    wins = row.get("externalWins", 0)
    losses = row.get("externalLosses", 0)
    latest = row.get("latestEpoch") or {}
    initial = row.get("latestInitialPolicyShift") or {}
    result["chains"][name] = {
        "generation": row.get("generation"),
        "snapshotId": row.get("snapshotId"),
        "completedShards": row.get("completedShards"),
        "episodes": row.get("episodes"),
        "decisions": row.get("decisions"),
        "externalWins": wins,
        "externalLosses": losses,
        "externalWinRate": wins / (wins + losses) if wins + losses else None,
        "selfPlayEpisodes": row.get("selfPlayEpisodes"),
        "livePpoOpponentEpisodes": row.get("livePpoOpponentEpisodes"),
        "publishedUpdates": row.get("publishedUpdates"),
        "failedUpdates": row.get("failedUpdates"),
        "latestBatchDecisions": latest.get("decisions") or initial.get("decisions"),
        "initialKl": initial.get("approximateKl"),
        "latestKl": latest.get("approximateKl"),
        "latestClip": latest.get("clipFraction"),
    }
print(json.dumps(result, indent=2))
