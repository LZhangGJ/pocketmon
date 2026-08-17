#!/usr/bin/env bash
set -Eeuo pipefail

main=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
capacity=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/capacity-comparison
screen=$capacity/screening
worktree=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
python=/homes/lzhang/mypath/new/envs/trans/bin/python
runner=/homes/lzhang/run_latest_ppo_full_eval.py
run_shard=/homes/lzhang/run_isolated_arena_shard.sh
hosts=10.113.13.53,10.113.13.54,10.113.13.57,10.113.13.63,10.113.13.64,10.113.13.67,10.113.13.68,10.113.13.69,10.113.13.71,10.113.13.72,10.113.13.73,10.113.13.74,10.113.13.75,10.113.13.77,10.113.13.78

prepare_root() {
  local root=$1
  mkdir -p "$root/state"
  cp -a "$main/state/." "$root/state/"
}

run_stage() {
  local name=$1 root=$2 portable=$3 frozen_games=$4 shards=$5
  if [[ -s "$root/monitoring/full-matrix/latest.json" ]] && grep -q '"status": "complete"' "$root/monitoring/full-matrix/latest.json"; then
    echo "BC_SCREEN_SKIP_COMPLETE stage=$name at=$(date -Iseconds)"
    return
  fi
  prepare_root "$root"
  echo "BC_SCREEN_START stage=$name at=$(date -Iseconds)"
  PYTHONNOUSERSITE=1 "$python" -s "$runner" \
    --league-root "$root" --worktree "$worktree" --python "$python" \
    --run-shard "$run_shard" --bc-portable "$portable" \
    --games-per-frozen "$frozen_games" --games-per-head-to-head 20 --shards "$shards" \
    --distributed-hosts "$hosts" --max-shards-per-host 3 \
    >"$screen/logs/$name.log" 2>&1
  echo "BC_SCREEN_DONE stage=$name at=$(date -Iseconds)"
}

run_all() {
  mkdir -p "$screen/logs"
  while pgrep -f "run_latest_ppo_full_eval.py --league-root $main" >/dev/null; do
    echo "BC_SCREEN_WAIT_MAIN at=$(date -Iseconds)"
    sleep 20
  done
  run_stage standard_smoke "$screen/standard-smoke" "$capacity/standard_1m/universal_bc.npz" 2 16
  run_stage standard_frozen40 "$screen/standard-frozen40" "$capacity/standard_1m/universal_bc.npz" 40 45
  run_stage large_smoke "$screen/large-smoke" "$capacity/large_256x6/universal_bc.npz" 2 16
  run_stage large_frozen40 "$screen/large-frozen40" "$capacity/large_256x6/universal_bc.npz" 40 45
  touch "$screen/COMPLETE"
}

if [[ "${1:-}" == --run ]]; then run_all; exit; fi
mkdir -p "$screen"
pidfile="$screen/controller.pid"
if [[ -s "$pidfile" ]]; then
  old=$(<"$pidfile")
  if [[ "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "BC_SCREEN_ALREADY_RUNNING pid=$old"
    exit
  fi
fi
nohup /bin/bash "$0" --run >"$screen/controller.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
sleep 2
kill -0 "$pid"
echo "BC_SCREEN_CONTROLLER_STARTED pid=$pid"
