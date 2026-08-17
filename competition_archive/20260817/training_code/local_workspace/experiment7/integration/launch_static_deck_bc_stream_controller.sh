#!/usr/bin/env bash
set -euo pipefail

root=/dev/shm/lzhang-static-deck-bc-10d-20260815-build
script=/dataT0/Free/lzhang/pocketmon-runs/experiment7-static-deck-bc-10d-20260815/control/runtime/integration/stream_static_deck_bc_d17_to_workers.sh
mkdir -p "$root/stream-control"
if pgrep -af '[s]tream_static_deck_bc_d17_to_workers.sh' >/dev/null; then
  pgrep -af '[s]tream_static_deck_bc_d17_to_workers.sh'
  exit 0
fi
nohup bash "$script" >"$root/stream-control/controller.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$root/stream-control/controller.pid"
printf '%s\n' "$pid"
