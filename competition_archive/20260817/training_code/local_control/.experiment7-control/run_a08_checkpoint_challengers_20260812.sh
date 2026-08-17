#!/usr/bin/env bash
set -Eeuo pipefail

live_root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-checkpoint-challengers-20260812
worktree=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
python=/homes/lzhang/mypath/new/envs/trans/bin/python
runner=/homes/lzhang/run_latest_ppo_full_eval.py
run_shard=/homes/lzhang/run_isolated_arena_shard.sh
bc=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/final_7d/downstream-e3cb2936afb5/seed-20260812/universal_bc.npz
hosts=10.113.13.53,10.113.13.54,10.113.13.57,10.113.13.63,10.113.13.64,10.113.13.67,10.113.13.68,10.113.13.69,10.113.13.71,10.113.13.72,10.113.13.73,10.113.13.74,10.113.13.75,10.113.13.77,10.113.13.78

run_all() {
  mkdir -p "$root/logs"
  for generation in 253 277 294 333; do
    candidate=$(printf '%s/g%06d' "$root" "$generation")
    latest="$candidate/monitoring/full-matrix/latest.json"
    if [[ -s "$latest" ]] && grep -q '"status": "complete"' "$latest"; then
      echo "A08_CHECKPOINT_SKIP_COMPLETE generation=$generation"
      continue
    fi
    echo "A08_CHECKPOINT_START generation=$generation at=$(date -Iseconds)"
    PYTHONNOUSERSITE=1 "$python" -s "$runner" \
      --league-root "$candidate" \
      --worktree "$worktree" \
      --python "$python" \
      --run-shard "$run_shard" \
      --bc-portable "$bc" \
      --games-per-frozen 40 \
      --games-per-head-to-head 40 \
      --shards 45 \
      --distributed-hosts "$hosts" \
      --max-shards-per-host 3 \
      >"$root/logs/g$(printf '%06d' "$generation").log" 2>&1
    echo "A08_CHECKPOINT_DONE generation=$generation at=$(date -Iseconds)"
  done
  touch "$root/ALL_EVALUATIONS_COMPLETE"
}

if [[ "${1:-}" == "--run" ]]; then
  run_all
  exit
fi

mkdir -p "$root"
pidfile="$root/controller.pid"
if [[ -s "$pidfile" ]]; then
  old_pid=$(<"$pidfile")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "A08_CHECKPOINT_CONTROLLER_ALREADY_RUNNING pid=$old_pid"
    exit 0
  fi
fi
nohup /bin/bash "$0" --run >"$root/controller.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
sleep 2
kill -0 "$pid"
echo "A08_CHECKPOINT_CONTROLLER_STARTED pid=$pid root=$root"
