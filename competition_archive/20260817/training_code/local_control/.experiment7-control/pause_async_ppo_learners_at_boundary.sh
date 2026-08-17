#!/usr/bin/env bash
set -Eeuo pipefail

worktree="${1:?worktree is required}"
shift
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811

for chain in "$@"; do
  pid=$(pgrep -f "$worktree/experiment7/integration/run_async_ppo_learner.py.*--chain $chain" | head -1 || true)
  if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    echo "LEARNER_NOT_FOUND chain=$chain" >&2
    exit 2
  fi
  deadline=$((SECONDS + 600))
  while kill -0 "$pid" 2>/dev/null; do
    children=$(pgrep -P "$pid" || true)
    if [[ -z "$children" ]]; then
      kill -STOP "$pid"
      children=$(pgrep -P "$pid" || true)
      if [[ -z "$children" ]]; then
        kill -TERM "$pid"
        kill -CONT "$pid" 2>/dev/null || true
        for _ in {1..50}; do
          kill -0 "$pid" 2>/dev/null || break
          sleep 0.1
        done
        kill -0 "$pid" 2>/dev/null && { echo "LEARNER_DID_NOT_STOP chain=$chain pid=$pid" >&2; exit 3; }
        break
      fi
      kill -CONT "$pid"
    fi
    (( SECONDS < deadline )) || { echo "LEARNER_BOUNDARY_TIMEOUT chain=$chain pid=$pid" >&2; exit 4; }
    sleep 0.05
  done
  printf 'chain=%s old_pid=%s stopped_at=%s reason=parameter_adjustment_boundary\n' \
    "$chain" "$pid" "$(date -Iseconds)" >"$root/state/paused-$chain.txt"
  echo "LEARNER_PAUSED_AT_BOUNDARY chain=$chain pid=$pid"
done
