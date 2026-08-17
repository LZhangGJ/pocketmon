#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 || $# % 3 != 0 )); then
  echo "usage: $0 PARENT_PID CHILD_PID CHAIN [...]" >&2
  exit 2
fi

while (( $# )); do
  parent=$1
  child=$2
  chain=$3
  shift 3

  [[ "$parent" =~ ^[0-9]+$ && "$child" =~ ^[0-9]+$ ]]
  parent_cmd=$(tr '\0' ' ' <"/proc/$parent/cmdline")
  child_cmd=$(tr '\0' ' ' <"/proc/$child/cmdline")
  [[ "$parent_cmd" == *run_async_ppo_learner.py*"--chain $chain"* ]]
  [[ "$child_cmd" == *train_universal_ppo.py* ]]

  kill -STOP "$parent"
  kill -TERM "$child"
  for _ in $(seq 1 60); do
    kill -0 "$child" 2>/dev/null || break
    sleep 0.5
  done
  if kill -0 "$child" 2>/dev/null; then
    echo "child did not exit after TERM: $child" >&2
    exit 3
  fi

  kill -TERM "$parent"
  kill -CONT "$parent" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$parent" 2>/dev/null || break
    sleep 0.25
  done
  if kill -0 "$parent" 2>/dev/null; then
    echo "parent did not exit after TERM: $parent" >&2
    exit 4
  fi
  printf 'STOPPED chain=%s parent=%s child=%s\n' "$chain" "$parent" "$child"
done
