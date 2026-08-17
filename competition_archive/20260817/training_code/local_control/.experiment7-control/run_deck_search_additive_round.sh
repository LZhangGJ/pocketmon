#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 12 ]]; then
  echo "usage: $0 OUTPUT_ROOT LEARNERS SOURCE_ROUND OPPONENTS PPO_ROOT WORKTREE EXPECTED_COMMIT PYTHON CG_DIR TARGET_GAMES COMPLETED_GAMES CANDIDATE_GAMES" >&2
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
target_games="${10}"
completed_games="${11}"
candidate_games="${12}"
controllers="$ppo_root/controllers"
shard_count=12
add_games=$((target_games - completed_games))

if [[ "$output_root" != /* ]] || [[ -e "$output_root" ]]; then
  echo "output root must be a new absolute path: $output_root" >&2
  exit 3
fi
if (( target_games <= completed_games || candidate_games < target_games )); then
  echo "expected candidate >= target > completed games" >&2
  exit 4
fi
if [[ "$(git -C "$worktree" rev-parse HEAD)" != "$expected_commit" ]] || [[ -n "$(git -C "$worktree" status --porcelain)" ]]; then
  echo "Arena worktree is not the expected clean commit" >&2
  exit 5
fi

mkdir -p "$output_root"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
cd "$worktree"
"$python" "$worktree/scripts/league_v1.py" \
  --snapshot "$opponents" \
  --schedule "$output_root/candidate_schedule${candidate_games}.csv" \
  --games-per-pair "$candidate_games" \
  --learners-manifest "$learners"

if [[ -f "$source_round/results.csv" ]]; then
  source_results=("$source_round/results.csv")
else
  source_results=("$source_round"/results-shard-*.csv)
  if (( ${#source_results[@]} != shard_count )); then
    echo "expected results.csv or 12 source result shards" >&2
    exit 6
  fi
fi
"$python" "$controllers/manage_additive_arena_round.py" prepare \
  --schedule "$output_root/candidate_schedule${candidate_games}.csv" \
  --learners "$learners" \
  --source-results "${source_results[@]}" \
  --output-dir "$output_root" \
  --target-games "$target_games" \
  --completed-games "$completed_games"

pids=()
for ((shard=0; shard<shard_count; shard++)); do
  output="$output_root/results-add${add_games}-shard-$shard.csv"
  log="$output_root/shard-$shard.log"
  bash "$controllers/run_isolated_arena_shard.sh" \
    "$worktree" "$python" "$output_root/pending_add${add_games}.csv" \
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
  echo "one or more additive Arena shards failed" >&2
  exit 7
fi

new_results=("$output_root"/results-add${add_games}-shard-*.csv)
"$python" "$controllers/manage_additive_arena_round.py" finalize \
  --schedule "$output_root/selected_schedule${target_games}.csv" \
  --completed "$output_root/completed_round${completed_games}.csv" \
  --new-results "${new_results[@]}" \
  --output "$output_root/results.csv"
sha256sum \
  "$output_root/candidate_schedule${candidate_games}.csv" \
  "$output_root/selected_schedule${target_games}.csv" \
  "$output_root/completed_round${completed_games}.csv" \
  "$output_root/pending_add${add_games}.csv" "$output_root/additive_receipt.json" \
  "${new_results[@]}" "$output_root/results.csv" >"$output_root/SHA256SUMS"
touch "$output_root/SUCCESS"
echo "DECK_ADDITIVE_ROUND_SUCCESS root=$output_root target_per_pair=$target_games added_per_pair=$add_games"
