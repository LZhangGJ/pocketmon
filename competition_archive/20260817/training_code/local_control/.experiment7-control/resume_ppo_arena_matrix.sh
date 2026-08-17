#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 ARENA_ROOT PPO_ROOT OPPONENTS_JSON WORKTREE PYTHON CG_DIR" >&2
  exit 2
fi

arena_root=$(realpath "$1")
ppo_root=$(realpath "$2")
opponents=$(realpath "$3")
worktree=$(realpath "$4")
python=$(realpath "$5")
cg_dir=$(realpath "$6")
controllers="$ppo_root/controllers"
schedule="$arena_root/schedule.csv"
learners="$arena_root/learners.json"
runtime="$arena_root/runtime"
shard_count=12

for required in \
  "$schedule" "$learners" "$arena_root/packages.json" \
  "$runtime/prepare_arena_runtime_receipt.json"; do
  if [[ ! -s "$required" ]]; then
    echo "required prepared Arena input is missing: $required" >&2
    exit 3
  fi
done
if [[ -e "$arena_root/SUCCESS" ]] || [[ -e "$arena_root/summary" ]]; then
  echo "Arena is already complete or summarized: $arena_root" >&2
  exit 4
fi
for ((shard=0; shard<shard_count; shard++)); do
  if [[ -e "$arena_root/results-shard-$shard.csv" ]]; then
    echo "refusing to overwrite existing shard result: $shard" >&2
    exit 5
  fi
done

pids=()
for ((shard=0; shard<shard_count; shard++)); do
  shard_root="$runtime/runtime-shard-$shard"
  output="$arena_root/results-shard-$shard.csv"
  log="$arena_root/retry1-shard-$shard.log"
  if [[ -e "$log" ]]; then
    echo "refusing to overwrite retry log: $log" >&2
    exit 6
  fi
  bash "$controllers/run_isolated_arena_shard.sh" \
    "$worktree" "$python" "$schedule" \
    "$shard_root/learners.json" "$shard_root/opponents.json" \
    "$cg_dir" "$output" "$shard" "$shard_count" "$arena_root" \
    >"$log" 2>&1 &
  pid=$!
  echo "$pid" >"$arena_root/retry1-shard-$shard.pid"
  pids+=("$pid")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "one or more retry1 Arena shards failed" >&2
  exit 7
fi

results=("$arena_root"/results-shard-*.csv)
if (( ${#results[@]} != shard_count )); then
  echo "unexpected result shard count: ${#results[@]} != $shard_count" >&2
  exit 8
fi
"$python" "$controllers/summarize_arena_matrix.py" \
  --schedule "$schedule" \
  --results "${results[@]}" \
  --learners "$learners" \
  --opponents "$opponents" \
  --output-dir "$arena_root/summary"
"$python" "$controllers/select_ppo_pair.py" \
  --summary "$arena_root/summary/summary.json" \
  --learners "$learners" \
  --output "$arena_root/selection.json"
sha256sum \
  "$arena_root/packages.json" "$learners" "$arena_root/learners_receipt.json" \
  "$schedule" "$runtime/prepare_arena_runtime_receipt.json" \
  "$arena_root/summary/summary.json" "$arena_root/summary/ranking.csv" \
  "$arena_root/summary/payoff_matrix.csv" "$arena_root/selection.json" \
  >"$arena_root/SHA256SUMS"
touch "$arena_root/SUCCESS"
echo "PPO_ARENA_RETRY1_SUCCESS root=$arena_root games=$((12 * 11 * 20))"
