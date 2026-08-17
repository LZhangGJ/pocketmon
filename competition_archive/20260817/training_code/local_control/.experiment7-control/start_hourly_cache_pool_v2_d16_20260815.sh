#!/usr/bin/env bash
set -euo pipefail

source_root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
eval_league_root=/suedata1/Free/lzhang/pocketmon-runs/experiment7-hourly-cache-pool-distributed-v2-20260815
hourly_root="$eval_league_root/monitoring/hourly-cache-pool"
state_dir="$eval_league_root/state"
source_league="$source_root/state/league.json"
driver="$eval_league_root/control/run_hourly_cache_pool_eval_v3_20260815.py"
worktree=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
python=/homes/lzhang/mypath/new/envs/trans/bin/python3.11
run_shard=/homes/lzhang/run_isolated_arena_shard.sh
bc_portable=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/final_7d/downstream-e3cb2936afb5/seed-20260812/universal_bc.npz

for required in "$source_league" "$driver" "$worktree" "$python" "$run_shard" "$bc_portable"; do
  [[ -e "$required" ]] || { echo "MISSING_REQUIRED_PATH $required" >&2; exit 2; }
done

if pgrep -af "$driver.*--league-root $eval_league_root" >/dev/null; then
  echo "HOURLY_CACHE_POOL_EVAL_ALREADY_RUNNING"
  pgrep -af "$driver.*--league-root $eval_league_root"
  exit 0
fi

mkdir -p "$state_dir" "$hourly_root" "$eval_league_root/monitoring"
[[ -e "$state_dir/league.json" ]] || ln -s "$source_league" "$state_dir/league.json"
[[ -e "$eval_league_root/monitoring/full-matrix" ]] || ln -s "$hourly_root" "$eval_league_root/monitoring/full-matrix"

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
log="$hourly_root/manual-$timestamp.log"
nohup "$python" -s "$driver" \
  --league-root "$eval_league_root" \
  --worktree "$worktree" \
  --python "$python" \
  --run-shard "$run_shard" \
  --bc-portable "$bc_portable" \
  --games-per-frozen 2 \
  --games-per-head-to-head 20 \
  --shards 36 \
  --distributed-hosts 10.113.13.72,10.113.13.73,10.113.13.74 \
  --max-shards-per-host 12 \
  >"$log" 2>&1 </dev/null &

pid=$!
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo "HOURLY_CACHE_POOL_EVAL_FAILED_TO_STAY_RUNNING pid=$pid log=$log" >&2
  tail -80 "$log" >&2 || true
  exit 5
fi
echo "HOURLY_CACHE_POOL_EVAL_STARTED pid=$pid"
echo "LOG $log"
echo "OUTPUT $hourly_root"
