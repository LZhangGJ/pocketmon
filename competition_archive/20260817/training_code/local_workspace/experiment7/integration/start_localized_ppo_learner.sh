#!/usr/bin/env bash
set -euo pipefail

chain=${1:?chain}
device=${2:?cuda device}
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
local_root=/dev/shm/experiment7-ppo-local-runtime
python=/dev/shm/experiment7-ppo-python-env/bin/python
pidfile="$root/workers/learner-$chain.pid"
log="$root/logs/learner-$chain-localized.log"

test -f /dev/shm/experiment7-ppo-python-env/SUCCESS
test -f "$local_root/experiment7/integration/run_async_ppo_learner.py"
test -d "$local_root/experiment7/reference"

if [[ -s "$pidfile" ]]; then
  old_pid=$(<"$pidfile")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "existing learner pid=$old_pid chain=$chain" >&2
    exit 3
  fi
fi

nohup env PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$python" -s "$local_root/experiment7/integration/run_async_ppo_learner.py" \
    --league "$root/state/league.json" \
    --chain "$chain" \
    --worktree "$local_root" \
    --reference-root-override "$local_root/experiment7/reference" \
    --run-root "$root/learners" \
    --buffer-root "$root/buffer" \
    --python "$python" \
    --deployment-staging-root /dev/shm/experiment7-ppo-deploy-staging \
    --device "$device" \
    --max-behavior-lag 2 \
    --min-decisions 12000 \
    --teacher-anchor-coefficient 0.05 \
    --seat1-weight 1.25 \
    --normalize-advantages-by-player \
    --balance-player-minibatches \
    --bootstrap-deployment \
    --poll-seconds 10 \
    >>"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
sleep 1
kill -0 "$pid"
printf '{"at":"%s","host":"%s","chain":"%s","device":"%s","pid":%s,"runtime":"localized"}\n' \
  "$(date -u +%FT%TZ)" "$(hostname)" "$chain" "$device" "$pid" \
  >"$root/control/large-g9-pool-reconfiguration-20260814/learner-transitions/$chain-localized.json"
echo "LOCALIZED_LEARNER_STARTED chain=$chain pid=$pid device=$device"
