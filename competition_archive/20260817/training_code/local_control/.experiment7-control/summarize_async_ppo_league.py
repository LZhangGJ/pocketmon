from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


league = read(ROOT / "state/league.json")
pool = read(Path(league["poolPath"]))
historical_live_archetypes = {}
for historical_chain in league["chains"].values():
    for snapshot in [*historical_chain.get("history", []), historical_chain.get("current", {})]:
        manifest_path = snapshot.get("packageManifest")
        if not manifest_path:
            continue
        try:
            manifest = read(Path(manifest_path))
        except (OSError, json.JSONDecodeError):
            continue
        for package in manifest.get("packages", []):
            historical_live_archetypes[str(package["name"])] = str(historical_chain["archetypeId"])
result = {
    "leagueUpdatedAt": league["updatedAt"],
    "poolSha256": league["poolSha256"],
    "liveAgents": [
        {
            "name": row["name"],
            "chain": row["ppo_chain"],
            "generation": row["ppo_generation"],
            "archetype": row["canonical_archetype"],
        }
        for row in pool["agents"]
        if row.get("pool_status") == "live_async_ppo_snapshot"
    ],
    "canonicalArchetypeCounts": dict(
        sorted(Counter(str(row["canonical_archetype"]) for row in pool["agents"]).items())
    ),
    "chains": {},
    "workerPidFiles": sorted(path.stem for path in (ROOT / "workers").glob("*.pid")),
    "legacyStopReceipts": sorted(path.name for path in (ROOT / "state").glob("legacy-*-stopped.txt")),
}
for chain_name, chain in sorted(league["chains"].items()):
    summaries = []
    for path in (ROOT / "buffer/ready" / chain_name).glob("*.jsonl.gz.summary.json"):
        try:
            summaries.append(read(path))
        except (OSError, json.JSONDecodeError):
            pass
    published = sorted((ROOT / "learners" / chain_name).glob("generation-*/PUBLISHED.json"))
    failed = sorted((ROOT / "learners" / chain_name).glob("generation-*/FAILED.json"))
    latest_metrics = None
    opponent_counts: Counter[str] = Counter()
    for summary in summaries:
        opponent_counts.update(
            {str(name): int(count) for name, count in summary.get("opponents", {}).items()}
        )
    pool_by_name = {str(row["name"]): row for row in pool["agents"]}
    opponent_archetypes: Counter[str] = Counter()
    for name, count in opponent_counts.items():
        if name == "self_play":
            continue
        if name in pool_by_name:
            archetype = str(pool_by_name[name]["canonical_archetype"])
        elif name in historical_live_archetypes:
            archetype = historical_live_archetypes[name]
        else:
            archetype = "UNKNOWN_RETIRED_AGENT"
        opponent_archetypes[archetype] += count
    metrics_path = Path(chain["current"]["checkpoint"]).with_name("metrics.json")
    if metrics_path.is_file():
        latest_metrics = read(metrics_path)
    result["chains"][chain_name] = {
        "generation": chain["current"]["generation"],
        "snapshotId": chain["current"]["snapshotId"],
        "completedShards": len(summaries),
        "episodes": sum(int(row.get("episodes", 0)) for row in summaries),
        "decisions": sum(int(row.get("decisions", 0)) for row in summaries),
        "externalWins": sum(int(row.get("wins", 0)) for row in summaries),
        "externalLosses": sum(int(row.get("losses", 0)) for row in summaries),
        "selfPlayEpisodes": sum(int(row.get("selfPlayEpisodes", 0)) for row in summaries),
        "livePpoOpponentEpisodes": sum(
            count for name, count in opponent_counts.items() if name.startswith("live_")
        ),
        "livePpoOpponents": dict(
            sorted((name, count) for name, count in opponent_counts.items() if name.startswith("live_"))
        ),
        "opponentArchetypeEpisodes": dict(sorted(opponent_archetypes.items())),
        "publishedUpdates": len(published),
        "failedUpdates": len(failed),
        "latestInitialPolicyShift": (
            latest_metrics.get("initialPolicyShift") if latest_metrics else None
        ),
        "latestEpoch": (
            latest_metrics.get("epochs", [])[-1]
            if latest_metrics and latest_metrics.get("epochs")
            else None
        ),
    }
print(json.dumps(result, indent=2, ensure_ascii=False))
