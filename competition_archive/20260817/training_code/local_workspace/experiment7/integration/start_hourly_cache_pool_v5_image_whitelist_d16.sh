#!/usr/bin/env bash
set -euo pipefail

source_root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
eval_league_root=/suedata1/Free/lzhang/pocketmon-runs/experiment7-hourly-image-whitelist-v5-20260816
hourly_root="$eval_league_root/monitoring/hourly-cache-pool"
state_dir="$eval_league_root/state"
driver="$eval_league_root/control/run_hourly_cache_pool_eval_v5_image_whitelist.py"
worktree=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
python=/homes/lzhang/mypath/new/envs/trans/bin/python3.11

if pgrep -af 'run_hourly_cache_pool_eval_v[345].*--league-root' >/dev/null; then
  echo "HOURLY_CACHE_POOL_EVAL_ALREADY_RUNNING"
  pgrep -af 'run_hourly_cache_pool_eval_v[345].*--league-root'
  exit 0
fi

for required in "$source_root/state/league.json" "$driver" "$worktree" "$python"; do
  [[ -e "$required" ]] || { echo "MISSING_REQUIRED_PATH $required" >&2; exit 2; }
done

mkdir -p "$hourly_root" "$state_dir" "$eval_league_root/monitoring"
[[ -e "$state_dir/league.json" ]] || ln -s "$source_root/state/league.json" "$state_dir/league.json"
[[ -e "$eval_league_root/monitoring/full-matrix" ]] || ln -s "$hourly_root" "$eval_league_root/monitoring/full-matrix"
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
log="$hourly_root/manual-$timestamp-image-whitelist-v5-d16.log"
nohup "$python" -s "$driver" \
  --league-root "$eval_league_root" \
  --worktree "$worktree" \
  --python "$python" \
  --run-shard "$eval_league_root/control/run_isolated_arena_shard_v5_retry.sh" \
  --bc-portable /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/final_7d/downstream-e3cb2936afb5/seed-20260812/universal_bc.npz \
  --games-per-frozen 2 \
  --games-per-head-to-head 20 \
  --timeout-seconds 600 \
  --timeout-retries 2 \
  --shards 24 \
  --distributed-hosts 10.113.13.74 \
  --max-shards-per-host 24 \
  >"$log" 2>&1 </dev/null &

pid=$!
sleep 5
kill -0 "$pid" 2>/dev/null || { tail -100 "$log" >&2; exit 5; }
echo "HOURLY_CACHE_POOL_IMAGE_WHITELIST_V5_STARTED pid=$pid"
echo "LOG $log"
