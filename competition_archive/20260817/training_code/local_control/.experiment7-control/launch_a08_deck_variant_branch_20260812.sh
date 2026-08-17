#!/usr/bin/env bash
set -Eeuo pipefail

: "${BRANCH:?BRANCH is required}"
: "${DECK:?DECK is required}"
: "${GPU_INDEX:?GPU_INDEX is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${INITIAL_CHECKPOINT:?INITIAL_CHECKPOINT is required}"
: "${TARGET_POOL:?TARGET_POOL is required}"

runner=${RUNNER:-/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-deck-branches-20260812/control/run_a08_deck_variant_branch_20260812.sh}
mkdir -p "$RUN_ROOT/control" "$RUN_ROOT/logs" "$RUN_ROOT/workers"
pidfile="$RUN_ROOT/workers/$BRANCH.pid"
log="$RUN_ROOT/logs/$BRANCH.log"

if [[ -s "$pidfile" ]]; then
  old_pid=$(<"$pidfile")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "BRANCH_ALREADY_RUNNING branch=$BRANCH pid=$old_pid"
    exit 0
  fi
fi

nohup env \
  BRANCH="$BRANCH" DECK="$DECK" GPU_INDEX="$GPU_INDEX" RUN_ROOT="$RUN_ROOT" \
  INITIAL_CHECKPOINT="$INITIAL_CHECKPOINT" TARGET_POOL="$TARGET_POOL" \
  /bin/bash "$runner" >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
sleep 3
kill -0 "$pid"
echo "BRANCH_STARTED branch=$BRANCH pid=$pid gpu=$GPU_INDEX log=$log"
