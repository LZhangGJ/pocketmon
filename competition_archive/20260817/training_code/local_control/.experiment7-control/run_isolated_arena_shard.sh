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

if [[ "$output" != /* ]]; then
  echo "output must be absolute: $output" >&2
  exit 2
fi
output_parent=$(realpath "$(dirname "$output")")
case "$output_parent/" in
  "$writable_root/"*) ;;
  *)
    echo "output is outside writable root: $output" >&2
    exit 2
    ;;
esac
# Learner and opponent manifests are frozen read-only inputs.  They do not
# need to be copied below the output root: bwrap's root bind is read-only and
# only the explicit writable root is rebound writable.  Keep the path gate on
# output, where crossing the boundary could mutate a frozen artifact.
if [[ -e "$output" ]]; then
  echo "refusing to overwrite Arena output: $output" >&2
  exit 3
fi

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
  "$python" "$worktree/scripts/run_league_schedule.py" \
    --schedule "$schedule" \
    --learners "$learners" \
    --opponents "$opponents" \
    --cg-dir "$cg_dir" \
    --output "$output" \
    --shard-index "$shard_index" \
    --shard-count "$shard_count" \
    --max-decisions 5000 \
    --timeout-seconds 180
