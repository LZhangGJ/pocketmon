#!/usr/bin/env python3
import json
import os
import subprocess
from pathlib import Path

league = Path(os.environ.get("LEAGUE", "/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811"))

def load(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception as exc:
        return {"_error": str(exc), "_path": str(path)}

out = {}
for name, path in {
    "full": league / "monitoring/full-matrix/latest.json",
    "submission4": league / "monitoring/submission4/latest.json",
}.items():
    payload = load(path)
    if name == "full" and "chains" in payload:
        out[name] = {k: payload.get(k) for k in ("status", "busy", "updatedAt", "roundId", "games", "engineSeedControlled", "frozenAgentCount")}
        out[name]["chains"] = {}
        for chain, row in payload["chains"].items():
            out[name]["chains"][chain] = {
                k: row.get(k) for k in (
                    "generation", "frozenAggregate", "universalBcFrozenAggregate", "ppoMinusBc",
                    "deltaVsPrevious", "progress", "seatMetrics", "seatGap",
                    "directVsUniversalBc", "ppoHeadToHead"
                )
            }
            out[name]["chains"][chain]["agents"] = [
                [a["agent"], a["ppo"]["scoreRate"], a["universalBc"]["scoreRate"], a["ppoMinusBc"], a["deltaVsPrevious"]]
                for a in row.get("agents", [])
            ]
    elif name == "submission4" and "current" in payload:
        out[name] = {k: payload.get(k) for k in ("status", "busy", "updatedAt", "roundId", "gamesPerAgent", "current", "cumulative", "completedRounds")}
        out[name]["generations"] = {row["chain"]: row["generation"] for row in payload.get("selected", [])}
    else:
        out[name] = payload

try:
    raw = subprocess.check_output([
        "/homes/lzhang/mypath/new/envs/trans/bin/python", "-s",
        "/homes/lzhang/summarize_async_ppo_league.py", "--league-root", str(league)
    ], text=True, timeout=240)
    training = json.loads(raw)
    keys = ("generation", "completedShards", "episodes", "decisions", "externalWins", "externalLosses",
            "selfPlayEpisodes", "livePpoOpponentEpisodes", "opponentArchetypeEpisodes", "publishedUpdates",
            "failedUpdates", "latestInitialPolicyShift", "latestEpoch")
    out["training"] = {
        "leagueUpdatedAt": training.get("leagueUpdatedAt"),
        "workers": len(training.get("workerPidFiles", [])),
        "legacyStopReceipts": training.get("legacyStopReceipts", []),
        "chains": {name: {key: row.get(key) for key in keys} for name, row in training["chains"].items()},
    }
except Exception as exc:
    out["training"] = {"_error": str(exc)}

refresh = Path("/dataT0/Free/lzhang/pocketmon-runs/replay-refresh-20260812")
out["replay"] = {
    "audits": {p.stem: load(p) for p in sorted((refresh / "audits").glob("*.json"))},
    "cacheManifests": {},
    "logs": {},
}
for date in ("2026-08-09", "2026-08-10", "2026-08-11"):
    path = refresh / "cache" / date / "prepared" / "universal_training_sources.json"
    if path.exists():
        data = load(path)
        summary = data.get("dataset", {}).get("summary", {})
        out["replay"]["cacheManifests"][date] = {
            "episodes": summary.get("sourceEpisodes"),
            "decisions": summary.get("decisions"),
            "trainDecisions": summary.get("trainDecisions"),
            "validationDecisions": summary.get("validationDecisions"),
        }
    log = refresh / "logs" / f"cache-{date}.log"
    if log.exists():
        raw = log.read_text(errors="replace").replace("\x00", "")
        out["replay"]["logs"][date] = raw.splitlines()[-8:]

inc = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812")
out["incremental"] = {}
for rel in ("controller.log", "capacity-comparison/controller.log", "capacity-comparison/control/exit-status.json"):
    p = inc / rel
    if p.exists():
        if p.suffix == ".json":
            out["incremental"][rel] = load(p)
        else:
            out["incremental"][rel] = p.read_text(errors="replace").splitlines()[-10:]
for candidate in ("standard_1m", "large_256x6"):
    report = inc / "capacity-comparison" / candidate / "training_report.json"
    if report.exists():
        out["incremental"][candidate] = load(report)
    log = inc / "capacity-comparison" / "logs" / f"{candidate}.log"
    if log.exists():
        out["incremental"][f"{candidate}.log"] = log.read_text(errors="replace").splitlines()[-12:]
    candidate_root = inc / "capacity-comparison" / candidate
    out["incremental"][f"{candidate}.files"] = [
        str(p.relative_to(candidate_root)) for p in sorted(candidate_root.rglob("*"))
        if p.is_file() and p.stat().st_size < 1_000_000
    ][-20:]

branch_root = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-deck-branches-20260812")
out["a08Variants"] = {}
for branch in ("a08_maxbelt", "a08_lilligant", "a08_lilligant_maxbelt"):
    rows = []
    for metrics in sorted((branch_root / branch).glob("generation-*/metrics.json")):
        rows.append(load(metrics))
    log = branch_root / "logs" / f"{branch}.log"
    out["a08Variants"][branch] = {
        "generations": len(rows),
        "latest": rows[-1] if rows else None,
        "logTail": log.read_text(errors="replace").splitlines()[-6:] if log.exists() else [],
    }

out["targeted"] = {}
for p in (league / "logs" / "worker-a08-targeted-doraemon17.log",):
    if p.is_file() and p.stat().st_size < 2_000_000:
        raw = p.read_text(errors="replace")
        out["targeted"][str(p.relative_to(league))] = raw.splitlines()[-12:]

targeted_summaries = sorted(league.rglob("a08-targeted-*.summary.json"))
targeted_rows = [load(p) for p in targeted_summaries]
opponent_counts = {}
for row in targeted_rows:
    for key in ("opponentSelectionCounts", "opponentCounts", "opponents"):
        value = row.get(key)
        if isinstance(value, dict):
            for opponent, count in value.items():
                if isinstance(count, (int, float)):
                    opponent_counts[opponent] = opponent_counts.get(opponent, 0) + int(count)
out["targetedSummary"] = {
    "files": len(targeted_summaries),
    "episodes": sum(int(r.get("episodes", 0)) for r in targeted_rows),
    "decisions": sum(int(r.get("decisions", 0)) for r in targeted_rows),
    "wins": sum(int(r.get("wins", 0)) for r in targeted_rows),
    "losses": sum(int(r.get("losses", 0)) for r in targeted_rows),
    "failures": sum(int(r.get("failures", 0)) for r in targeted_rows),
    "opponents": opponent_counts,
    "latest": targeted_rows[-1] if targeted_rows else None,
}

print(json.dumps(out, ensure_ascii=False))
