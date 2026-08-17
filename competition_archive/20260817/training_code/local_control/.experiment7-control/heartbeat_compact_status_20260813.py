import json
import subprocess


ROOT = "/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811"
PYTHON = "/homes/lzhang/mypath/new/envs/trans/bin/python"
SUMMARIZER = "/homes/lzhang/summarize_async_ppo_league.py"


raw = subprocess.check_output(
    [PYTHON, "-s", SUMMARIZER, "--league-root", ROOT],
    text=True,
)
data = json.loads(raw)
out = {}
for name, row in data.get("chains", {}).items():
    latest = row.get("latestEpoch") or {}
    out[name] = {
        "generation": row.get("generation"),
        "shards": row.get("completedShards"),
        "episodes": row.get("episodes") or row.get("totalEpisodes"),
        "decisions": row.get("decisions") or row.get("totalDecisions"),
        "externalWins": row.get("externalWins"),
        "externalLosses": row.get("externalLosses"),
        "selfPlay": row.get("selfPlayEpisodes"),
        "ppoOpponents": row.get("ppoOpponentEpisodes"),
        "published": row.get("publishedUpdates"),
        "failed": row.get("failedUpdates"),
        "kl": latest.get("approximateKl"),
        "clip": latest.get("clipFraction"),
        "ppoOpponentEpisodes": row.get("livePpoOpponentEpisodes"),
        "workers": row.get("workers"),
        "coverage": {k: v for k, v in (row.get("opponentArchetypeEpisodes") or {}).items() if k in {"ARCHALUDON_CINDERACE", "A06", "A01"}},
    }
out["_chainKeys"] = sorted(next(iter(data.get("chains", {}).values()), {}).keys())
print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
