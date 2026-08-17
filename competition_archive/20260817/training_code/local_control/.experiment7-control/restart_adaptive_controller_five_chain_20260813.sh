#!/usr/bin/env bash
set -euo pipefail

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
mapfile -t pids < <(pgrep -f '[a]daptive_ppo_training_controller.py' || true)
for pid in "${pids[@]}"; do
  kill "$pid"
done
for _ in $(seq 1 20); do
  if ! pgrep -f '[a]daptive_ppo_training_controller.py' >/dev/null; then
    break
  fi
  sleep 0.25
done
if pgrep -f '[a]daptive_ppo_training_controller.py' >/dev/null; then
  echo ADAPTIVE_CONTROLLER_OLD_PROCESS_DID_NOT_STOP >&2
  exit 2
fi
mkdir -p "$root/monitoring/adaptive-training"
nohup /homes/lzhang/mypath/new/envs/trans/bin/python -s \
  "$root/control/adaptive_ppo_training_controller.py" \
  --league "$root/state/league.json" \
  --matrix "$root/monitoring/full-matrix/latest.json" \
  --state "$root/state/adaptive-training-state.json" \
  --poll-seconds 30 \
  >"$root/monitoring/adaptive-training/controller.log" 2>&1 </dev/null &
echo "ADAPTIVE_CONTROLLER_FIVE_CHAIN_STARTED pid=$!"
