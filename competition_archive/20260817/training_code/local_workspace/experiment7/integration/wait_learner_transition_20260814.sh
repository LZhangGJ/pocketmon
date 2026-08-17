#!/usr/bin/env bash
set -u

mode=${1:?mode}
chain=${2:?chain}
device=${3:-cuda:0}
timeout_seconds=${4:-1800}
started=$SECONDS
while (( SECONDS - started < timeout_seconds )); do
  bash /homes/lzhang/transition_ppo_learner_20260814.sh "$mode" "$chain" "$device"
  status=$?
  (( status == 0 )) && exit 0
  (( status == 4 )) || exit "$status"
  sleep 10
done
echo "TRANSITION_TIMEOUT chain=$chain mode=$mode" >&2
exit 6
