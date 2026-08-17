#!/usr/bin/env bash
set -euo pipefail

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
control="$root/control"
source_script=/dev/shm/run_ppo_frozen_promotion_loop.sh
target_script="$control/run_ppo_frozen_promotion_loop.sh"

install -m 0755 "$source_script" "$target_script"

pid_file="$control/ppo-frozen-promotion-controller.pid"
if [[ -s "$pid_file" ]]; then
  pid=$(<"$pid_file")
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "ALREADY_RUNNING=$pid"
    exit 0
  fi
fi

log="$root/monitoring/ppo-frozen-promotion/controller.log"
nohup env PYTHON_BIN=/homes/lzhang/mypath/new/envs/trans/bin/python \
  PROMOTION_POLL_SECONDS=1800 \
  "$target_script" "$root" >>"$log" 2>&1 &
pid=$!
temporary="$pid_file.$pid.tmp"
printf '%s\n' "$pid" >"$temporary"
mv -f "$temporary" "$pid_file"
printf 'STARTED=%s\n' "$pid"
