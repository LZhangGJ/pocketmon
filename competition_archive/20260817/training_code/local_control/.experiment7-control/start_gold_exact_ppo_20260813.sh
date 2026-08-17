#!/usr/bin/env bash
set -Eeuo pipefail

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
chain=lucario_gold_exact
python=/homes/lzhang/mypath/new/envs/trans/bin/python
worktree=/homes/lzhang/worktrees/experiment7-a08-learner-clean-8885a59
launcher=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
pidfile="$root/workers/learner-$chain-doraemon13.pid"
log="$root/logs/learner-$chain-doraemon13.log"

mkdir -p "$root/workers" "$root/logs" "$root/monitoring/lucario-gold-exact"
if pgrep -f "run_async_ppo_learner.py .*--chain $chain( |$)" >/dev/null; then
  echo "GOLD_EXACT_LEARNER_ALREADY_RUNNING"
  pgrep -af "run_async_ppo_learner.py .*--chain $chain( |$)"
  exit 0
fi

nohup env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$python" "$launcher/experiment7/integration/run_async_ppo_learner.py" \
    --league "$root/state/league.json" --chain "$chain" --worktree "$worktree" \
    --run-root "$root/learners" --buffer-root "$root/buffer" --python "$python" \
    --deployment-staging-root /dev/shm/experiment7-ppo-deploy-staging \
    --device cuda:0 --max-behavior-lag 2 --min-decisions 8000 \
    --teacher-anchor-coefficient 0.04 --seat1-weight 1.0 \
    --normalize-advantages-by-player --balance-player-minibatches \
    --bootstrap-deployment --poll-seconds 10 \
    >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
sleep 4
kill -0 "$pid"
printf '{"status":"running","chain":"%s","host":"%s","gpu":0,"pid":%s,"startedAt":"%s"}\n' \
  "$chain" "$(hostname -s)" "$pid" "$(date -Iseconds)" \
  >"$root/monitoring/lucario-gold-exact/learner.json"
echo "GOLD_EXACT_LEARNER_STARTED pid=$pid host=$(hostname -s) gpu=0"
