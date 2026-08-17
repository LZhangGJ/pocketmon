#!/usr/bin/env bash
set -Eeuo pipefail

: "${BRANCH:?}"
: "${DECK:?}"
: "${GPU_INDEX:?}"
: "${EXPECTED_PID:?}"

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-deck-branches-20260812
runner="$root/control/run_a08_deck_variant_branch_20260812.sh"
launcher="$root/control/launch_a08_deck_variant_branch_20260812.sh"
pidfile="$root/workers/$BRANCH.pid"
log="$root/logs/$BRANCH.log"
initial=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/learners/a08_dipplin_seaking/generation-000307/checkpoint.pt
target="$root/control/a08_deck_variant_target_pool_20260812.json"

actual=$(<"$pidfile")
[[ "$actual" == "$EXPECTED_PID" ]] || { echo "PID_MISMATCH actual=$actual expected=$EXPECTED_PID"; exit 2; }
cmd=$(tr '\0' ' ' <"/proc/$actual/cmdline")
[[ "$cmd" == *"run_a08_deck_variant_branch_20260812.sh"* ]] || { echo "CMD_MISMATCH $cmd"; exit 3; }

cp -p "$log" "$log.pre-nproc-all-fix" 2>/dev/null || true
kill "$actual"
for _ in $(seq 1 20); do
  kill -0 "$actual" 2>/dev/null || break
  sleep 1
done
kill -0 "$actual" 2>/dev/null && { echo "OLD_PROCESS_STILL_RUNNING pid=$actual"; exit 4; }

BRANCH="$BRANCH" DECK="$DECK" GPU_INDEX="$GPU_INDEX" RUN_ROOT="$root" \
INITIAL_CHECKPOINT="$initial" TARGET_POOL="$target" RUNNER="$runner" \
  /bin/bash "$launcher"
