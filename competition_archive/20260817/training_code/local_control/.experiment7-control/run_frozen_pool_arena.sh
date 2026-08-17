#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 ARENA_ROOT PPO_ROOT OPPONENTS_JSON WORKTREE PYTHON CG_DIR GAMES_PER_PAIR" >&2
  exit 2
fi

arena_root=$(realpath "$1")
ppo_root=$(realpath "$2")
opponents=$(realpath "$3")
worktree=$(realpath "$4")
python=$(realpath "$5")
cg_dir=$(realpath "$6")
games_per_pair="$7"
controllers="$ppo_root/controllers"
packages="$arena_root/packages.json"
learners="$arena_root/learners.json"
receipt="$arena_root/learners_receipt.json"
schedule="$arena_root/schedule.csv"
runtime="$arena_root/runtime"
shard_count=12

if [[ ! "$games_per_pair" =~ ^[0-9]+$ ]] || (( games_per_pair < 1 )); then
  echo "games per pair must be positive" >&2
  exit 3
fi
for required in "$packages" "$learners" "$receipt" "$opponents"; do
  if [[ ! -s "$required" ]]; then
    echo "required frozen Arena input is missing: $required" >&2
    exit 4
  fi
done
for output in "$schedule" "$runtime" "$arena_root/summary" "$arena_root/SUCCESS"; do
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite Arena output: $output" >&2
    exit 5
  fi
done

cd "$worktree"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$python" scripts/league_v1.py \
  --snapshot "$opponents" \
  --schedule "$schedule" \
  --games-per-pair "$games_per_pair" \
  --learners-manifest "$learners"
"$python" experiment7/integration/stage_opponent_pool.py prepare-arena-runtime \
  --packages "$packages" \
  --opponents "$opponents" \
  --arena-stage "$runtime" \
  --shard-count "$shard_count"

pids=()
for ((shard=0; shard<shard_count; shard++)); do
  shard_root="$runtime/runtime-shard-$shard"
  output="$arena_root/results-shard-$shard.csv"
  log="$arena_root/shard-$shard.log"
  bash "$controllers/run_isolated_arena_shard.sh" \
    "$worktree" "$python" "$schedule" \
    "$shard_root/learners.json" "$shard_root/opponents.json" \
    "$cg_dir" "$output" "$shard" "$shard_count" "$arena_root" \
    >"$log" 2>&1 &
  pid=$!
  echo "$pid" >"$arena_root/shard-$shard.pid"
  pids+=("$pid")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "one or more frozen-pool Arena shards failed" >&2
  exit 6
fi

results=("$arena_root"/results-shard-*.csv)
if (( ${#results[@]} != shard_count )); then
  echo "unexpected result shard count: ${#results[@]}" >&2
  exit 7
fi
"$python" "$controllers/summarize_arena_matrix.py" \
  --schedule "$schedule" \
  --results "${results[@]}" \
  --learners "$learners" \
  --opponents "$opponents" \
  --output-dir "$arena_root/summary"
sha256sum \
  "$packages" "$learners" "$receipt" "$schedule" \
  "$runtime/prepare_arena_runtime_receipt.json" \
  "$arena_root/summary/summary.json" "$arena_root/summary/ranking.csv" \
  "$arena_root/summary/payoff_matrix.csv" >"$arena_root/SHA256SUMS"
touch "$arena_root/SUCCESS"
learner_count=$("$python" -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["agents"]))' "$learners")
opponent_count=$("$python" -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["agents"]))' "$opponents")
echo "FROZEN_POOL_ARENA_SUCCESS root=$arena_root games=$((learner_count * opponent_count * games_per_pair))"
