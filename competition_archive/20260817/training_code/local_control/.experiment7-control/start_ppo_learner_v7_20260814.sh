#!/usr/bin/env bash
set -Eeuo pipefail

chain="${1:?chain required}"
device="${2:?CUDA device required}"
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
launcher=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
train_tree=/homes/lzhang/worktrees/experiment7-a08-learner-clean-8885a59
python=/homes/lzhang/mypath/new/envs/trans/bin/python
pidfile="$root/workers/learner-$chain.pid"
log="$root/logs/learner-$chain.log"
mkdir -p "$root/workers" "$root/logs" "$root/buffer" "$root/learners"

if [[ -s "$pidfile" ]]; then
  old_pid=$(<"$pidfile")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "LEARNER_ALREADY_RUNNING chain=$chain pid=$old_pid"
    exit 0
  fi
fi

nohup env PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  LD_PRELOAD=/homes/lzhang/mypath/new/envs/wsl4mis/lib/libstdc++.so.6 \
  "$python" -s "$launcher/experiment7/integration/run_async_ppo_learner.py" \
    --league "$root/state/league.json" \
    --chain "$chain" \
    --worktree "$train_tree" \
    --run-root "$root/learners" \
    --buffer-root "$root/buffer" \
    --python "$python" \
    --deployment-staging-root /dev/shm/experiment7-ppo-deploy-staging \
    --device "$device" \
    --max-behavior-lag 2 \
    --min-decisions 12000 \
    --teacher-anchor-coefficient 0.04 \
    --seat1-weight 1.25 \
    --normalize-advantages-by-player \
    --balance-player-minibatches \
    --bootstrap-deployment \
    --poll-seconds 10 \
    >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
sleep 1
kill -0 "$pid"
echo "LEARNER_STARTED chain=$chain pid=$pid device=$device log=$log"
