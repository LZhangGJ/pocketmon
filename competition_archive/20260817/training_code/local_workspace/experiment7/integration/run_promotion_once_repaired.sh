#!/usr/bin/env bash
set -euo pipefail

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
integration=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0/experiment7/integration
python_bin=/homes/lzhang/mypath/new/envs/trans/bin/python

PYTHONPATH="$integration" "$python_bin" -s "$root/control/promote_ppo_to_frozen_pool.py" \
  --league "$root/state/league.json" \
  --state "$root/control/ppo-frozen-promotion-state.json" \
  --source-base-pool "$root/state/opponent-pool-large-g9-133-plus-best-a02-a08.json" \
  --output-base-pool "$root/state/opponent-pool-large-g9-133-plus-best-a02-a08-promoted.json" \
  --reports-root "$root/monitoring/full-matrix" \
  --reports-root "$root/monitoring/ppo-frozen-promotion" \
  --min-frozen-games 40 \
  --min-direct-games 40 \
  --max-regression-pp 2 \
  --required-passes 2 \
  --tactical-evidence "$root/monitoring/ppo-tactical-guardrails/latest.json"
