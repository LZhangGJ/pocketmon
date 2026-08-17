#!/usr/bin/env bash
set -euo pipefail

mode=${1:?start|restart|stop}
chain=${2:?chain}
device=${3:-cuda:0}
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
launcher=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
train_tree=/homes/lzhang/worktrees/experiment7-a08-learner-clean-8885a59
python=/homes/lzhang/mypath/new/envs/trans/bin/python
pidfile="$root/workers/learner-$chain.pid"
log="$root/logs/learner-$chain.log"
receipt_dir="$root/control/large-g9-pool-reconfiguration-20260814/learner-transitions"
mkdir -p "$root/workers" "$root/logs" "$root/buffer" "$root/learners" "$receipt_dir"

old_pid=
if [[ -s "$pidfile" ]]; then
  old_pid=$(<"$pidfile")
fi
if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
  command_text=$(tr '\0' ' ' <"/proc/$old_pid/cmdline")
  [[ "$command_text" == *run_async_ppo_learner.py*"--chain $chain"* ]] || {
    echo "PIDFILE_COMMAND_MISMATCH pid=$old_pid chain=$chain" >&2
    exit 3
  }
  kill -STOP "$old_pid"
  children=$(pgrep -P "$old_pid" || true)
  if [[ -n "$children" ]]; then
    kill -CONT "$old_pid"
    echo "DEFERRED_ACTIVE_CHILD chain=$chain pid=$old_pid children=$children"
    exit 4
  fi
  kill -TERM "$old_pid"
  kill -CONT "$old_pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$old_pid" 2>/dev/null || break
    sleep 0.25
  done
  kill -0 "$old_pid" 2>/dev/null && {
    echo "OLD_LEARNER_DID_NOT_EXIT chain=$chain pid=$old_pid" >&2
    exit 5
  }
fi

if [[ "$mode" == stop ]]; then
  printf '{"at":"%s","host":"%s","chain":"%s","mode":"stop","oldPid":"%s"}\n' \
    "$(date -u +%FT%TZ)" "$(hostname)" "$chain" "$old_pid" \
    >"$receipt_dir/$chain-stop.json"
  echo "LEARNER_STOPPED chain=$chain oldPid=$old_pid"
  exit 0
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
    --teacher-anchor-coefficient 0.05 \
    --seat1-weight 1.25 \
    --normalize-advantages-by-player \
    --balance-player-minibatches \
    --bootstrap-deployment \
    --poll-seconds 10 \
    >>"$log" 2>&1 </dev/null &
new_pid=$!
printf '%s\n' "$new_pid" >"$pidfile"
sleep 1
kill -0 "$new_pid"
printf '{"at":"%s","host":"%s","chain":"%s","mode":"%s","device":"%s","oldPid":"%s","newPid":%s}\n' \
  "$(date -u +%FT%TZ)" "$(hostname)" "$chain" "$mode" "$device" "$old_pid" "$new_pid" \
  >"$receipt_dir/$chain-$mode.json"
echo "LEARNER_STARTED chain=$chain pid=$new_pid device=$device"
