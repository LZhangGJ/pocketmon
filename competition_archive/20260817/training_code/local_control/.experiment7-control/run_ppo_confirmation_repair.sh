#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 SOURCE_ROOT REPAIR_ROOT PPO_ROOT OPPONENTS WORKTREE EXPECTED_COMMIT PYTHON" >&2
  exit 2
fi

source_root=$(realpath "$1")
repair_root="$2"
ppo_root=$(realpath "$3")
opponents=$(realpath "$4")
worktree=$(realpath "$5")
expected_commit="$6"
python=$(realpath "$7")
controllers="$ppo_root/controllers"
cg_dir=/dataT0/Free/lzhang/pocketmon-runs/experiment7-opponent-pool-20260810/official-engine/cg
shard_count=12

if [[ "$repair_root" != /* ]] || [[ -e "$repair_root" ]]; then
  echo "repair root must be a new absolute path: $repair_root" >&2
  exit 3
fi
if [[ "$(git -C "$worktree" rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "worktree commit mismatch" >&2
  exit 4
fi
if [[ -n "$(git -C "$worktree" status --porcelain)" ]]; then
  echo "repair worktree is not clean" >&2
  exit 5
fi

export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"$python" "$controllers/prepare_ppo_confirmation_repair.py" \
  --source-root "$source_root" \
  --output-root "$repair_root" \
  --expect-failures 400

pids=()
for ((shard=0; shard<shard_count; shard++)); do
  output="$repair_root/results-repair-shard-$shard.csv"
  log="$repair_root/shard-$shard.log"
  bash "$controllers/run_isolated_arena_shard.sh" \
    "$worktree" "$python" "$repair_root/failed_schedule.csv" \
    "$repair_root/selected_learners.json" "$opponents" \
    "$cg_dir" "$output" "$shard" "$shard_count" "$repair_root" \
    >"$log" 2>&1 &
  pid=$!
  echo "$pid" >"$repair_root/shard-$shard.pid"
  pids+=("$pid")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "one or more confirmation repair shards failed" >&2
  exit 6
fi

repair_results=("$repair_root"/results-repair-shard-*.csv)
"$python" "$controllers/summarize_arena_matrix.py" \
  --schedule "$source_root/selected_schedule300.csv" \
  --results "$source_root/completed_screening100.csv" \
    "$repair_root/completed_add200_success.csv" "${repair_results[@]}" \
  --learners "$repair_root/selected_learners.json" \
  --opponents "$opponents" \
  --output-dir "$repair_root/summary"
"$python" "$controllers/select_ppo_pair.py" \
  --summary "$repair_root/summary/summary.json" \
  --learners "$repair_root/selected_learners.json" \
  --output "$repair_root/selection.json"

"$python" - "$repair_root/summary/ranking.csv" <<'PY'
import csv
import sys
from pathlib import Path

with Path(sys.argv[1]).open(encoding="utf-8-sig", newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 2 or any(int(row["games"]) != 3300 or int(row["failures"]) != 0 for row in rows):
    raise SystemExit(f"confirmation repair gate failed: {rows}")
PY

sha256sum \
  "$repair_root/failed_schedule.csv" "$repair_root/completed_add200_success.csv" \
  "$repair_root/selected_learners.json" "$repair_root/repair_receipt.json" \
  "${repair_results[@]}" "$repair_root/summary/summary.json" \
  "$repair_root/summary/ranking.csv" "$repair_root/summary/payoff_matrix.csv" \
  "$repair_root/selection.json" >"$repair_root/SHA256SUMS"
touch "$repair_root/SUCCESS"
echo "PPO_CONFIRMATION_REPAIR_SUCCESS root=$repair_root repaired=400 total=6600"
