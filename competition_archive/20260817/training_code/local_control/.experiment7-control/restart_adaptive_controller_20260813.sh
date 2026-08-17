#!/usr/bin/env bash
set -Eeuo pipefail

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
pidfile="$root/workers/adaptive-training-controller.pid"
old_pid=""
if [[ -s "$pidfile" ]]; then
  old_pid=$(<"$pidfile")
fi
if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
  kill "$old_pid"
  for _ in 1 2 3 4 5; do
    kill -0 "$old_pid" 2>/dev/null || break
    sleep 1
  done
fi
if pgrep -f '[a]daptive_ppo_training_controller.py' >/dev/null; then
  echo "DUPLICATE_CONTROLLER_REMAINS" >&2
  pgrep -af '[a]daptive_ppo_training_controller.py' >&2
  exit 2
fi
nohup env PYTHONNOUSERSITE=1 \
  /homes/lzhang/mypath/new/envs/trans/bin/python -s \
  "$root/control/adaptive_ppo_training_controller.py" \
  --league "$root/state/league.json" \
  --matrix "$root/monitoring/full-matrix/latest.json" \
  --state "$root/state/adaptive-training-state.json" \
  --poll-seconds 30 \
  >"$root/logs/adaptive-training-controller.log" 2>&1 </dev/null &
new_pid=$!
printf '%s\n' "$new_pid" >"$pidfile"
sleep 3
kill -0 "$new_pid"
echo "CONTROLLER_RESTARTED pid=$new_pid"
