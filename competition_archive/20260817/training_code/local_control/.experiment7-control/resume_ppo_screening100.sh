#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 SCREEN_ROOT SMOKE_ROOT PPO_ROOT OPPONENTS_JSON WORKTREE PYTHON CG_DIR" >&2
  exit 2
fi

screen_root=$(realpath "$1")
smoke_root=$(realpath "$2")
ppo_root=$(realpath "$3")
opponents=$(realpath "$4")
worktree=$(realpath "$5")
python=$(realpath "$6")
cg_dir=$(realpath "$7")
controllers="$ppo_root/controllers"
runtime="$screen_root/runtime"
shard_count=12

for required in \
  "$screen_root/selected_schedule100.csv" "$screen_root/completed_smoke20.csv" \
  "$screen_root/pending_add80.csv" "$screen_root/selected_learners.json" \
  "$screen_root/screening_receipt.json" "$smoke_root/packages.json"; do
  if [[ ! -s "$required" ]]; then
    echo "required prepared screening input is missing: $required" >&2
    exit 3
  fi
done
if [[ -e "$screen_root/SUCCESS" ]] || [[ -e "$screen_root/summary" ]]; then
  echo "screening is already complete or summarized: $screen_root" >&2
  exit 4
fi
for ((shard=0; shard<shard_count; shard++)); do
  if [[ -e "$screen_root/results-add80-shard-$shard.csv" ]]; then
    echo "refusing to overwrite existing screening result: $shard" >&2
    exit 5
  fi
done

cd "$worktree"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
if [[ ! -s "$runtime/prepare_arena_runtime_receipt.json" ]]; then
  "$python" experiment7/integration/stage_opponent_pool.py prepare-arena-runtime \
    --packages "$smoke_root/packages.json" \
    --opponents "$opponents" \
    --arena-stage "$runtime" \
    --shard-count "$shard_count"
fi

pids=()
for ((shard=0; shard<shard_count; shard++)); do
  shard_root="$runtime/runtime-shard-$shard"
  output="$screen_root/results-add80-shard-$shard.csv"
  log="$screen_root/retry1-shard-$shard.log"
  if [[ -e "$log" ]]; then
    echo "refusing to overwrite retry1 log: $log" >&2
    exit 6
  fi
  bash "$controllers/run_isolated_arena_shard.sh" \
    "$worktree" "$python" "$screen_root/pending_add80.csv" \
    "$shard_root/learners.json" "$shard_root/opponents.json" \
    "$cg_dir" "$output" "$shard" "$shard_count" "$screen_root" \
    >"$log" 2>&1 &
  pid=$!
  echo "$pid" >"$screen_root/retry1-shard-$shard.pid"
  pids+=("$pid")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "one or more retry1 100-game screening shards failed" >&2
  exit 7
fi

new_results=("$screen_root"/results-add80-shard-*.csv)
if (( ${#new_results[@]} != shard_count )); then
  echo "unexpected add80 result shard count: ${#new_results[@]}" >&2
  exit 8
fi
"$python" "$controllers/summarize_arena_matrix.py" \
  --schedule "$screen_root/selected_schedule100.csv" \
  --results "$screen_root/completed_smoke20.csv" "${new_results[@]}" \
  --learners "$screen_root/selected_learners.json" \
  --opponents "$opponents" \
  --output-dir "$screen_root/summary"
"$python" "$controllers/select_ppo_pair.py" \
  --summary "$screen_root/summary/summary.json" \
  --learners "$screen_root/selected_learners.json" \
  --output "$screen_root/selection.json"
sha256sum \
  "$screen_root/full_schedule12x100.csv" "$screen_root/selected_learners.json" \
  "$screen_root/selected_schedule100.csv" "$screen_root/completed_smoke20.csv" \
  "$screen_root/pending_add80.csv" "$screen_root/screening_receipt.json" \
  "$runtime/prepare_arena_runtime_receipt.json" \
  "$screen_root/summary/summary.json" "$screen_root/summary/ranking.csv" \
  "$screen_root/summary/payoff_matrix.csv" "$screen_root/selection.json" \
  >"$screen_root/SHA256SUMS"
touch "$screen_root/SUCCESS"
echo "PPO_SCREENING100_RETRY1_SUCCESS root=$screen_root games=$((4 * 11 * 100)) new=$((4 * 11 * 80))"
