#!/usr/bin/env bash
set -euo pipefail

control=/dataT0/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/control
mkdir -p "$control"
if pgrep -af watch_bc_admission_then_increment_20260813.py | grep -v grep >/dev/null; then
  echo BC_INCREMENT_WATCHER_ALREADY_RUNNING
  exit 0
fi
nohup /homes/lzhang/mypath/new/envs/trans/bin/python -s \
  /homes/lzhang/watch_bc_admission_then_increment_20260813.py \
  >"$control/8-11-admission-to-8-12-increment.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$control/8-11-admission-to-8-12-increment.pid"
echo "BC_INCREMENT_WATCHER_STARTED pid=$pid"
