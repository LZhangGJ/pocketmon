import glob
import json
import os
import sys


ROOT = sys.argv[1] if len(sys.argv) > 1 else "/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-deck-branch-evals-20260813/arch-extra"
out = {}
for path in sorted(glob.glob(f"{ROOT}/*/monitoring/full-matrix/latest.json")):
    data = json.load(open(path, encoding="utf-8"))
    name = os.path.basename(path.split("/monitoring/")[0])
    chain = next(iter(data.get("chains", {}).values()), {})
    frozen = chain.get("frozenAggregate") or {}
    bc_frozen = chain.get("universalBcFrozenAggregate") or {}
    direct = chain.get("directVsUniversalBc") or {}
    seats = chain.get("seatMetrics") or {}
    arch = next((x for x in chain.get("agents", []) if x.get("agent") == "public_archaludon_meta"), {})
    out[name] = {
        "status": data.get("status"),
        "games": data.get("games"),
        "agentCount": data.get("frozenAgentCount"),
        "failures": frozen.get("failures"),
        "ppoWins": frozen.get("wins"),
        "ppoLosses": frozen.get("losses"),
        "ppoScoreRate": frozen.get("scoreRate"),
        "bcScoreRate": bc_frozen.get("scoreRate"),
        "seat0": seats.get("0", {}).get("scoreRate"),
        "seat1": seats.get("1", {}).get("scoreRate"),
        "directBC": direct.get("scoreRate"),
        "archPpo": arch.get("ppo", {}).get("scoreRate"),
        "archBc": arch.get("universalBc", {}).get("scoreRate"),
    }
print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
