#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 CONFIRM_ROOT SMOKE_ROOT SCREEN_ROOT PPO_ROOT OPPONENTS_JSON WORKTREE PYTHON" >&2
  exit 2
fi

confirm_root="$1"
smoke_root=$(realpath "$2")
screen_root=$(realpath "$3")
ppo_root=$(realpath "$4")
opponents=$(realpath "$5")
worktree=$(realpath "$6")
python=$(realpath "$7")
controllers="$ppo_root/controllers"
cg_dir=/dataT0/Free/lzhang/pocketmon-runs/experiment7-opponent-pool-20260810/official-engine/cg
shard_count=12
if [[ "$confirm_root" != /* ]] || [[ -e "$confirm_root" ]]; then
  echo "confirmation root must be a new absolute path: $confirm_root" >&2
  exit 3
fi

mkdir -p "$confirm_root"
cd "$worktree"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
"$python" scripts/league_v1.py \
  --snapshot "$opponents" \
  --schedule "$confirm_root/full_schedule12x300.csv" \
  --games-per-pair 300 \
  --learners-manifest "$smoke_root/learners.json"

screen_results=("$screen_root"/results-add80-shard-*.csv)
if (( ${#screen_results[@]} != shard_count )); then
  echo "unexpected screening result shard count: ${#screen_results[@]}" >&2
  exit 4
fi
"$python" "$controllers/prepare_ppo_confirmation.py" \
  --selection "$screen_root/selection.json" \
  --learners "$smoke_root/learners.json" \
  --schedule "$confirm_root/full_schedule12x300.csv" \
  --results "$screen_root/completed_smoke20.csv" "${screen_results[@]}" \
  --output-dir "$confirm_root"

pids=()
for ((shard=0; shard<shard_count; shard++)); do
  output="$confirm_root/results-add200-shard-$shard.csv"
  log="$confirm_root/shard-$shard.log"
  bash "$controllers/run_isolated_arena_shard.sh" \
    "$worktree" "$python" "$confirm_root/pending_add200.csv" \
    "$confirm_root/selected_learners.json" "$opponents" \
    "$cg_dir" "$output" "$shard" "$shard_count" "$confirm_root" \
    >"$log" 2>&1 &
  pid=$!
  echo "$pid" >"$confirm_root/shard-$shard.pid"
  pids+=("$pid")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "one or more 300-game confirmation shards failed" >&2
  exit 5
fi
new_results=("$confirm_root"/results-add200-shard-*.csv)
"$python" "$controllers/summarize_arena_matrix.py" \
  --schedule "$confirm_root/selected_schedule300.csv" \
  --results "$confirm_root/completed_screening100.csv" "${new_results[@]}" \
  --learners "$confirm_root/selected_learners.json" \
  --opponents "$opponents" \
  --output-dir "$confirm_root/summary"
"$python" "$controllers/select_ppo_pair.py" \
  --summary "$confirm_root/summary/summary.json" \
  --learners "$confirm_root/selected_learners.json" \
  --output "$confirm_root/selection.json"
sha256sum \
  "$confirm_root/full_schedule12x300.csv" "$confirm_root/selected_learners.json" \
  "$confirm_root/selected_schedule300.csv" "$confirm_root/completed_screening100.csv" \
  "$confirm_root/pending_add200.csv" "$confirm_root/confirmation_receipt.json" \
  "$confirm_root/summary/summary.json" "$confirm_root/summary/ranking.csv" \
  "$confirm_root/summary/payoff_matrix.csv" "$confirm_root/selection.json" \
  >"$confirm_root/SHA256SUMS"
touch "$confirm_root/SUCCESS"
echo "PPO_CONFIRMATION300_SUCCESS root=$confirm_root games=$((2 * 11 * 300)) new=$((2 * 11 * 200))"
