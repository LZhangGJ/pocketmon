#!/usr/bin/env bash
set -Eeuo pipefail

staging=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/monitoring/selected-best-vs-frozen-100/rounds/.20260811T144710Z-a05g80-a08g45-submission4.in-progress
eval_root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/monitoring/selected-best-vs-frozen-100
worktree=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
python=/homes/lzhang/mypath/new/envs/trans/bin/python
guard=/homes/lzhang/run_load_guarded_arena_shard.sh
cg=/dataT0/Free/lzhang/pocketmon-runs/experiment7-opponent-pool-20260810/official-engine/cg

test -d "$staging"
test -x "$guard"

run_missing() {
  local shard="$1"
  local output
  output=$(printf '%s/raw/results-shard-%03d.csv' "$staging" "$shard")
  if [[ -s "$output" ]]; then
    echo "ALREADY_COMPLETE shard=$shard output=$output"
    return 0
  fi
  "$guard" \
    "$worktree" "$python" "$staging/schedule.csv" \
    "$staging/learners.json" "$staging/opponents.json" \
    "$cg" "$output" "$shard" 24 "$eval_root"
}

run_missing 11 >"$staging/logs/resume-shard-011-doraemon02.log" 2>&1 &
pid11=$!
run_missing 23 >"$staging/logs/resume-shard-023-doraemon02.log" 2>&1 &
pid23=$!
failed=0
wait "$pid11" || failed=1
wait "$pid23" || failed=1
if (( failed )); then
  echo "RESUME_FAILED" >&2
  exit 1
fi
echo "RESUME_COMPLETE shards=11,23"
