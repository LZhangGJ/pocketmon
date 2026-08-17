#!/usr/bin/env bash
set -euo pipefail

source_root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
eval_league_root=/suedata1/Free/lzhang/pocketmon-runs/experiment7-hourly-cache-pool-distributed
hourly_root="$eval_league_root/monitoring/hourly-cache-pool"
bc_cache_source=/suedata1/Free/lzhang/pocketmon-runs/experiment7-hourly-cache-pool/monitoring/hourly-cache-pool/universal-bc-baseline
state_dir="$eval_league_root/state"
source_league="$source_root/state/league.json"
driver="$eval_league_root/control/run_hourly_cache_pool_eval_20260814.py"
worktree=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
python=/homes/lzhang/mypath/new/envs/trans/bin/python3.11
run_shard=/homes/lzhang/run_isolated_arena_shard.sh
bc_portable=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/final_7d/downstream-e3cb2936afb5/seed-20260812/universal_bc.npz

for required in "$source_league" "$driver" "$worktree" "$python" "$run_shard" "$bc_portable"; do
  if [[ ! -e "$required" ]]; then
    echo "MISSING_REQUIRED_PATH $required" >&2
    exit 2
  fi
done

if pgrep -af "$driver.*--league-root $eval_league_root" >/dev/null; then
  echo "HOURLY_CACHE_POOL_EVAL_ALREADY_RUNNING"
  pgrep -af "$driver.*--league-root $eval_league_root"
  exit 0
fi

mkdir -p "$state_dir" "$hourly_root"

if [[ -L "$hourly_root/universal-bc-baseline" ]]; then
  if [[ "$(readlink -f "$hourly_root/universal-bc-baseline")" != "$(readlink -f "$bc_cache_source")" ]]; then
    echo "UNEXPECTED_BC_CACHE_SYMLINK $hourly_root/universal-bc-baseline" >&2
    exit 3
  fi
elif [[ -e "$hourly_root/universal-bc-baseline" ]]; then
  echo "REFUSING_TO_REPLACE_EXISTING $hourly_root/universal-bc-baseline" >&2
  exit 3
elif [[ -d "$bc_cache_source" ]]; then
  ln -s "$bc_cache_source" "$hourly_root/universal-bc-baseline"
fi

if [[ -L "$state_dir/league.json" ]]; then
  if [[ "$(readlink -f "$state_dir/league.json")" != "$(readlink -f "$source_league")" ]]; then
    echo "UNEXPECTED_LEAGUE_SYMLINK $state_dir/league.json" >&2
    exit 3
  fi
elif [[ -e "$state_dir/league.json" ]]; then
  echo "REFUSING_TO_REPLACE_EXISTING $state_dir/league.json" >&2
  exit 3
else
  ln -s "$source_league" "$state_dir/league.json"
fi

if [[ -L "$eval_league_root/monitoring/full-matrix" ]]; then
  if [[ "$(readlink -f "$eval_league_root/monitoring/full-matrix")" != "$(readlink -f "$hourly_root")" ]]; then
    echo "UNEXPECTED_OUTPUT_SYMLINK $eval_league_root/monitoring/full-matrix" >&2
    exit 4
  fi
elif [[ -e "$eval_league_root/monitoring/full-matrix" ]]; then
  echo "REFUSING_TO_REPLACE_EXISTING $eval_league_root/monitoring/full-matrix" >&2
  exit 4
else
  ln -s "$hourly_root" "$eval_league_root/monitoring/full-matrix"
fi

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
log="$hourly_root/manual-$timestamp.log"
nohup "$python" -s "$driver" \
  --league-root "$eval_league_root" \
  --worktree "$worktree" \
  --python "$python" \
  --run-shard "$run_shard" \
  --bc-portable "$bc_portable" \
  --games-per-frozen 4 \
  --games-per-head-to-head 20 \
  --shards 96 \
  --distributed-hosts 10.113.13.72,10.113.13.73,10.113.13.74,10.113.13.77 \
  --max-shards-per-host 24 \
  >"$log" 2>&1 </dev/null &

pid=$!
sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "HOURLY_CACHE_POOL_EVAL_FAILED_TO_STAY_RUNNING pid=$pid log=$log" >&2
  tail -80 "$log" >&2 || true
  exit 5
fi

echo "HOURLY_CACHE_POOL_EVAL_STARTED pid=$pid"
echo "LOG $log"
echo "OUTPUT $hourly_root"
