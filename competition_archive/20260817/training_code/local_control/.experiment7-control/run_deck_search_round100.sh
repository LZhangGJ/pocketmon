#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: $0 OUTPUT_ROOT LEARNERS SOURCE_ROUND20 OPPONENTS PPO_ROOT WORKTREE EXPECTED_COMMIT PYTHON CG_DIR" >&2
  exit 2
fi

output_root="$1"
learners=$(realpath "$2")
source_round=$(realpath "$3")
opponents=$(realpath "$4")
ppo_root=$(realpath "$5")
worktree=$(realpath "$6")
expected_commit="$7"
python=$(realpath "$8")
cg_dir=$(realpath "$9")
controllers="$ppo_root/controllers"
shard_count=12

if [[ "$output_root" != /* ]] || [[ -e "$output_root" ]]; then
  echo "output root must be a new absolute path: $output_root" >&2
  exit 3
fi
if [[ "$(git -C "$worktree" rev-parse HEAD)" != "$expected_commit" ]] || [[ -n "$(git -C "$worktree" status --porcelain)" ]]; then
  echo "round100 worktree is not the expected clean commit" >&2
  exit 4
fi

mkdir -p "$output_root"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cd "$worktree"
"$python" "$worktree/scripts/league_v1.py" \
  --snapshot "$opponents" \
  --schedule "$output_root/candidate_schedule120.csv" \
  --games-per-pair 120 \
  --learners-manifest "$learners"
source_results=("$source_round"/results-shard-*.csv)
if (( ${#source_results[@]} != shard_count )); then
  echo "expected 12 round20 result shards" >&2
  exit 5
fi
"$python" "$controllers/manage_additive_arena_round.py" prepare \
  --schedule "$output_root/candidate_schedule120.csv" \
  --learners "$learners" \
  --source-results "${source_results[@]}" \
  --output-dir "$output_root" \
  --target-games 100 \
  --completed-games 20

pids=()
for ((shard=0; shard<shard_count; shard++)); do
  output="$output_root/results-add80-shard-$shard.csv"
  log="$output_root/shard-$shard.log"
  bash "$controllers/run_isolated_arena_shard.sh" \
    "$worktree" "$python" "$output_root/pending_add80.csv" \
    "$learners" "$opponents" "$cg_dir" "$output" \
    "$shard" "$shard_count" "$output_root" >"$log" 2>&1 &
  pid=$!
  echo "$pid" >"$output_root/shard-$shard.pid"
  pids+=("$pid")
done
failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "one or more deck round100 shards failed" >&2
  exit 6
fi

new_results=("$output_root"/results-add80-shard-*.csv)
"$python" "$controllers/manage_additive_arena_round.py" finalize \
  --schedule "$output_root/selected_schedule100.csv" \
  --completed "$output_root/completed_round20.csv" \
  --new-results "${new_results[@]}" \
  --output "$output_root/results.csv"
sha256sum \
  "$output_root/candidate_schedule120.csv" "$output_root/selected_schedule100.csv" \
  "$output_root/completed_round20.csv" \
  "$output_root/pending_add80.csv" "$output_root/additive_receipt.json" \
  "${new_results[@]}" "$output_root/results.csv" >"$output_root/SHA256SUMS"
touch "$output_root/SUCCESS"
echo "DECK_ROUND100_SUCCESS root=$output_root games=6400 added=5120"
