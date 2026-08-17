#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 ARENA_ROOT PPO_ROOT DOWNSTREAM_ROOT OPPONENTS_JSON WORKTREE PYTHON CG_DIR" >&2
  exit 2
fi

arena_root="$1"
ppo_root=$(realpath "$2")
downstream=$(realpath "$3")
opponents=$(realpath "$4")
worktree=$(realpath "$5")
python=$(realpath "$6")
cg_dir=$(realpath "$7")
if [[ "$arena_root" != /* ]] || [[ -e "$arena_root" ]]; then
  echo "arena root must be a new absolute path: $arena_root" >&2
  exit 3
fi

controllers="$ppo_root/controllers"
portable_root="$ppo_root/portable-eval-653f7a1b9cbe"
diversity_g10="$ppo_root/portable-eval-725e57dda09a/diversity/g0010/packages/packages.json"
g0="$downstream/seed-20260812-stabletie-653f7a1b9cbe/packages/packages.json"
packages="$arena_root/packages.json"
learners="$arena_root/learners.json"
manifest_receipt="$arena_root/learners_receipt.json"
schedule="$arena_root/schedule.csv"
runtime="$arena_root/runtime"
shard_count=12

cd "$worktree"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$python" "$controllers/build_ppo_arena_manifest.py" \
  --entry generalist 0 "$g0" seed20260812_stabletie__02_a03_606a775392ff \
  --entry generalist 10 "$portable_root/generalist/g0010/packages/packages.json" generalist_g0010__02_a03_606a775392ff \
  --entry generalist 20 "$portable_root/generalist/g0020/packages/packages.json" generalist_g0020__02_a03_606a775392ff \
  --entry hard_exploiter 0 "$g0" seed20260812_stabletie__03_a02_cafa7652a634 \
  --entry hard_exploiter 10 "$portable_root/hard_exploiter/g0010/packages/packages.json" hard_exploiter_g0010__03_a02_cafa7652a634 \
  --entry hard_exploiter 20 "$portable_root/hard_exploiter/g0020/packages/packages.json" hard_exploiter_g0020__03_a02_cafa7652a634 \
  --entry diversity 0 "$g0" seed20260812_stabletie__04_a08_1a88a53fe3d0 \
  --entry diversity 10 "$diversity_g10" diversity_g0010__04_a08_1a88a53fe3d0 \
  --entry diversity 20 "$portable_root/diversity/g0020/packages/packages.json" diversity_g0020__04_a08_1a88a53fe3d0 \
  --entry conservative 0 "$g0" seed20260812_stabletie__05_a06_89e6155f2531 \
  --entry conservative 10 "$portable_root/conservative/g0010/packages/packages.json" conservative_g0010__05_a06_89e6155f2531 \
  --entry conservative 20 "$portable_root/conservative/g0020/packages/packages.json" conservative_g0020__05_a06_89e6155f2531 \
  --output "$packages" \
  --learners-output "$learners" \
  --receipt "$manifest_receipt"

"$python" scripts/league_v1.py \
  --snapshot "$opponents" \
  --schedule "$schedule" \
  --games-per-pair 20 \
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
  echo "one or more Arena shards failed" >&2
  exit 5
fi

results=("$arena_root"/results-shard-*.csv)
if (( ${#results[@]} != shard_count )); then
  echo "unexpected result shard count: ${#results[@]} != $shard_count" >&2
  exit 6
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
  "$packages" "$learners" "$manifest_receipt" "$schedule" \
  "$runtime/prepare_arena_runtime_receipt.json" \
  "$arena_root/summary/summary.json" "$arena_root/summary/ranking.csv" \
  "$arena_root/summary/payoff_matrix.csv" "$arena_root/selection.json" \
  >"$arena_root/SHA256SUMS"
touch "$arena_root/SUCCESS"
echo "PPO_ARENA_SUCCESS root=$arena_root games=$((12 * 11 * 20))"
