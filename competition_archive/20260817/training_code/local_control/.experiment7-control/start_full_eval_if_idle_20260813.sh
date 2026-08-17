#!/usr/bin/env bash
set -euo pipefail

if pgrep -af '[r]un_latest_ppo_full_eval.py' >/dev/null; then
  echo FULL_EVAL_ALREADY_RUNNING
  exit 0
fi
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
log="$root/monitoring/full-matrix/manual-confirm-a02-20260813.log"
nohup /homes/lzhang/mypath/new/envs/trans/bin/python -s /homes/lzhang/run_latest_ppo_full_eval.py \
  --league-root "$root" \
  --worktree /homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0 \
  --python /homes/lzhang/mypath/new/envs/trans/bin/python \
  --run-shard /homes/lzhang/run_isolated_arena_shard.sh \
  --bc-portable /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/final_7d/downstream-e3cb2936afb5/seed-20260812/universal_bc.npz \
  --games-per-frozen 4 --games-per-head-to-head 20 --shards 45 \
  --distributed-hosts 10.113.13.53,10.113.13.54,10.113.13.57,10.113.13.63,10.113.13.64,10.113.13.67,10.113.13.68,10.113.13.69,10.113.13.71,10.113.13.72,10.113.13.73,10.113.13.74,10.113.13.75,10.113.13.77,10.113.13.78 \
  --max-shards-per-host 3 >"$log" 2>&1 </dev/null &
echo "FULL_EVAL_STARTED pid=$! log=$log"
