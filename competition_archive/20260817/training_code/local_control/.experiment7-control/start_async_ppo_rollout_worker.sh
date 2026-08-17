#!/usr/bin/env bash
set -Eeuo pipefail

worker_id="${1:?worker id is required}"
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
worktree=/homes/lzhang/worktrees/experiment7-async-4c45f89
python=/homes/lzhang/mypath/new/envs/trans/bin/python
pidfile="$root/workers/$worker_id.pid"
log="$root/logs/worker-$worker_id.log"
mkdir -p "$root/workers" "$root/logs" "$root/buffer"

if [[ -s "$pidfile" ]]; then
  old_pid=$(<"$pidfile")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "WORKER_ALREADY_RUNNING worker=$worker_id pid=$old_pid"
    exit 0
  fi
fi

nohup env \
  PYTHONNOUSERSITE=1 \
  OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  LD_PRELOAD=/homes/lzhang/mypath/new/envs/wsl4mis/lib/libstdc++.so.6 \
  ionice -c 2 -n 7 nice -n 10 \
  "$python" "$worktree/experiment7/integration/run_async_ppo_rollout_worker.py" \
    --league "$root/state/league.json" \
    --worktree "$worktree" \
    --buffer-root "$root/buffer" \
    --python "$python" \
    --worker-id "$worker_id" \
    --episodes-per-shard 20 \
    --refresh-rounds 1 \
    --self-play-fraction 0.25 \
    >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
sleep 1
kill -0 "$pid"
echo "WORKER_STARTED worker=$worker_id pid=$pid log=$log"
