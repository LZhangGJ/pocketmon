#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
  echo "usage: $0 WORKTREE PYTHON SCHEDULE LEARNERS OPPONENTS CG_DIR OUTPUT SHARD_INDEX SHARD_COUNT WRITABLE_ROOT" >&2
  exit 2
fi

worktree=$(realpath "$1")
python=$(realpath "$2")
schedule=$(realpath "$3")
learners=$(realpath "$4")
opponents=$(realpath "$5")
cg_dir=$(realpath "$6")
output="$7"
shard_index="$8"
shard_count="$9"
writable_root=$(realpath "${10}")

output_parent=$(realpath "$(dirname "$output")")
case "$output_parent/" in
  "$writable_root/"*) ;;
  *) echo "output is outside writable root: $output" >&2; exit 2 ;;
esac
[[ ! -e "$output" ]] || { echo "refusing to overwrite Arena output: $output" >&2; exit 3; }

exec bwrap \
  --unshare-net \
  --ro-bind / / \
  --dev-bind /dev /dev \
  --proc /proc \
  --tmpfs /tmp \
  --bind "$writable_root" "$writable_root" \
  --chdir "$worktree" \
  --setenv OPENBLAS_NUM_THREADS 1 \
  --setenv OMP_NUM_THREADS 1 \
  --setenv MKL_NUM_THREADS 1 \
  --setenv LD_LIBRARY_PATH /homes/lzhang/mypath/new/lib:/homes/lzhang/cuda12.9/cuda-12.9/lib64 \
  "$python" "$worktree/scripts/run_league_schedule_v4_retry.py" \
    --schedule "$schedule" \
    --learners "$learners" \
    --opponents "$opponents" \
    --cg-dir "$cg_dir" \
    --output "$output" \
    --shard-index "$shard_index" \
    --shard-count "$shard_count" \
    --max-decisions 5000 \
    --timeout-seconds "${ARENA_TIMEOUT_SECONDS:-600}" \
    --timeout-retries "${ARENA_TIMEOUT_RETRIES:-2}"
