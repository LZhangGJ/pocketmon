#!/usr/bin/env bash
set -euo pipefail

main_root=${1:?main root is required}
loop="$main_root/control/run_ppo_frozen_promotion_loop.sh"
log="$main_root/monitoring/ppo-frozen-promotion-controller.log"
pidfile="$main_root/control/ppo-frozen-promotion-controller.pid"

if [[ -s "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
  printf 'ALREADY_RUNNING=%s\n' "$(cat "$pidfile")"
  exit 0
fi

chmod +x "$loop"
nohup "$loop" "$main_root" >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
printf 'CONTROLLER_PID=%s\n' "$pid"
