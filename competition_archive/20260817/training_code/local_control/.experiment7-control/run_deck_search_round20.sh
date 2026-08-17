#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 ROUND_ROOT PACKAGES OPPONENTS WORKTREE PYTHON CG_DIR CONTROLLERS SEED" >&2
  exit 2
fi

round_root="$1"
packages=$(realpath "$2")
opponents=$(realpath "$3")
worktree=$(realpath "$4")
python=$(realpath "$5")
cg_dir=$(realpath "$6")
controllers=$(realpath "$7")
seed="$8"
shard_count=12
if [[ "$round_root" != /* ]] || [[ -e "$round_root" ]]; then
  echo "round root must be a new absolute path: $round_root" >&2
  exit 3
fi

cd "$worktree"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$python" "$controllers/prepare_deck_search_round.py" \
  --packages "$packages" \
  --opponents "$opponents" \
  --opponent public_alakazam_search_v9 \
  --opponent team_grim_model_a \
  --opponent public_archaludon_meta \
  --opponent team_submission_4_portable_bc \
  --games-per-pair 20 \
  --seed "$seed" \
  --output-dir "$round_root"

pids=()
for ((shard=0; shard<shard_count; shard++)); do
  output="$round_root/results-shard-$shard.csv"
  log="$round_root/shard-$shard.log"
  bash "$controllers/run_isolated_arena_shard.sh" \
    "$worktree" "$python" "$round_root/schedule20.csv" \
    "$round_root/learners.json" "$round_root/opponents.json" \
    "$cg_dir" "$output" "$shard" "$shard_count" "$round_root" \
    >"$log" 2>&1 &
  pid=$!
  echo "$pid" >"$round_root/shard-$shard.pid"
  pids+=("$pid")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "one or more deck-search Arena shards failed" >&2
  exit 4
fi
results=("$round_root"/results-shard-*.csv)
if (( ${#results[@]} != shard_count )); then
  echo "unexpected deck-search result shard count: ${#results[@]}" >&2
  exit 5
fi
"$python" "$controllers/summarize_arena_matrix.py" \
  --schedule "$round_root/schedule20.csv" \
  --results "${results[@]}" \
  --learners "$round_root/learners.json" \
  --opponents "$round_root/opponents.json" \
  --output-dir "$round_root/summary"
sha256sum \
  "$round_root/receipt.json" "$round_root/learners.json" \
  "$round_root/opponents.json" "$round_root/schedule20.csv" \
  "$round_root/summary/summary.json" "$round_root/summary/ranking.csv" \
  "$round_root/summary/payoff_matrix.csv" >"$round_root/SHA256SUMS"
touch "$round_root/SUCCESS"
echo "DECK_SEARCH_ROUND20_SUCCESS root=$round_root games=$((64 * 4 * 20))"
