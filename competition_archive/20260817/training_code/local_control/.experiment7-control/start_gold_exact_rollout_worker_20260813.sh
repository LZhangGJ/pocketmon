#!/usr/bin/env bash
set -Eeuo pipefail

worker_id=${1:?worker id required}
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
chain=lucario_gold_exact
python=/homes/lzhang/mypath/new/envs/trans/bin/python
worktree=/homes/lzhang/worktrees/experiment7-async-4c45f89
pidfile="$root/workers/$worker_id.pid"
log="$root/logs/worker-$worker_id.log"

mkdir -p "$root/workers" "$root/logs"
if [[ -s "$pidfile" ]]; then
  old=$(<"$pidfile")
  if [[ "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "GOLD_EXACT_WORKER_ALREADY_RUNNING worker=$worker_id pid=$old"
    exit 0
  fi
fi

nohup env PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ionice -c2 -n7 nice -n10 \
  "$python" "$worktree/experiment7/integration/run_async_ppo_rollout_worker.py" \
    --league "$root/state/league.json" --worktree "$worktree" \
    --buffer-root "$root/buffer" --python "$python" --worker-id "$worker_id" \
    --episodes-per-shard 20 --refresh-rounds 1 --self-play-fraction 0.25 \
    --cpu-limit 95 --io-limit 80 --only-chain "$chain" \
    >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
sleep 2
kill -0 "$pid"
echo "GOLD_EXACT_WORKER_STARTED worker=$worker_id pid=$pid host=$(hostname -s)"
