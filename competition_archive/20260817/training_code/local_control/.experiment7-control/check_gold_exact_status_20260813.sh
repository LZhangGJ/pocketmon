#!/usr/bin/env bash
set -u

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
python3 - "$root" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
league = json.loads((root / "state" / "league.json").read_text())
current = league["chains"]["lucario_gold_exact"]["current"]
keys = ("generation", "checkpoint", "packageManifest", "packageAgentId")
print(json.dumps({key: current.get(key) for key in keys}, indent=2))
print("FROZEN_GOLD", [
    agent.get("id") for agent in league.get("frozenPool", [])
    if "lucario_gold_exact" in agent.get("id", "")
])
PY
pgrep -af run_async_ppo_learner.py | grep lucario_gold_exact || true
