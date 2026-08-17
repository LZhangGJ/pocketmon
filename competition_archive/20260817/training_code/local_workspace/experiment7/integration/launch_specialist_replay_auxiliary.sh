#!/usr/bin/env bash
set -euo pipefail

deck_key=${1:?deck key required}
log_path=${2:?log path required}
nohup bash /dev/shm/prepare_specialist_replay_auxiliary.sh "$deck_key" \
  >"$log_path" 2>&1 </dev/null &
pid=$!
disown "$pid" 2>/dev/null || true
echo "$pid"
