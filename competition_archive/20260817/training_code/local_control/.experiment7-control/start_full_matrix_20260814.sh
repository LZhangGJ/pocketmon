#!/usr/bin/env bash
set -euo pipefail

if pgrep -af '[r]un_latest_ppo_full_eval_20260814.py' >/dev/null; then
  echo FULL_EVAL_ALREADY_RUNNING
  exit 0
fi

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
control="$root/monitoring/full-matrix/control"
log="$root/monitoring/full-matrix/manual-latest-20260814.log"

nohup /homes/lzhang/mypath/new/envs/trans/bin/python3.11 -s \
  "$control/run_latest_ppo_full_eval_20260814.py" \
  --league-root "$root" \
  --worktree /homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0 \
  --python /homes/lzhang/mypath/new/envs/trans/bin/python3.11 \
  --run-shard /homes/lzhang/run_isolated_arena_shard.sh \
  --bc-portable /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/final_7d/downstream-e3cb2936afb5/seed-20260812/universal_bc.npz \
  --games-per-frozen 4 \
  --games-per-head-to-head 20 \
  --shards 24 \
  --distributed-hosts 10.113.13.57,10.113.13.63,10.113.13.64,10.113.13.67,10.113.13.69,10.113.13.71,10.113.13.54,10.113.13.73 \
  --max-shards-per-host 3 \
  >"$log" 2>&1 </dev/null &

echo "FULL_EVAL_STARTED pid=$! log=$log"
