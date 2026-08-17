#!/usr/bin/env python3
import json
import subprocess

command = [
    "/homes/lzhang/mypath/new/envs/trans/bin/python", "-s",
    "/homes/lzhang/summarize_async_ppo_league.py",
]
completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
marker = '{\n  "leagueUpdatedAt"'
start = completed.stdout.rfind(marker)
if start < 0:
    raise RuntimeError("summary JSON marker not found")
payload = json.loads(completed.stdout[start:])
result = {"leagueUpdatedAt": payload.get("leagueUpdatedAt"), "chains": {}}
for name, row in payload["chains"].items():
    shift = row.get("latestInitialPolicyShift") or {}
    epoch = row.get("latestEpoch") or {}
    wins = int(row.get("externalWins", 0))
    losses = int(row.get("externalLosses", 0))
    result["chains"][name] = {
        "generation": row.get("generation"),
        "completedShards": row.get("completedShards"),
        "episodes": row.get("episodes"),
        "decisions": row.get("decisions"),
        "externalWins": wins,
        "externalLosses": losses,
        "externalWinRate": wins / max(wins + losses, 1),
        "selfPlayEpisodes": row.get("selfPlayEpisodes"),
        "livePpoOpponentEpisodes": row.get("livePpoOpponentEpisodes"),
        "publishedUpdates": row.get("publishedUpdates"),
        "failedUpdates": row.get("failedUpdates"),
        "initialKl": shift.get("approximateKl"),
        "initialClip": shift.get("clipFraction"),
        "latestKl": epoch.get("approximateKl"),
        "latestClip": epoch.get("clipFraction"),
        "latestBatchDecisions": epoch.get("decisions"),
    }
print(json.dumps(result, indent=2))
