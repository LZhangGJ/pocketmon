#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 WORKTREE REPLAY_DIR ANALYSIS_DIR OUTPUT_ROOT [extra prepare args...]" >&2
  exit 2
fi

WORKTREE="$1"
REPLAY_DIR="$2"
ANALYSIS_DIR="$3"
OUTPUT_ROOT="$4"
shift 4
PYTHON="${PYTHON:-/homes/lzhang/mypath/new/envs/trans/bin/python}"

cd "$WORKTREE"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$OUTPUT_ROOT/audit"
"$PYTHON" experiment7/integration/prepare_multideck_data.py \
  --replay-dir "$REPLAY_DIR" \
  --analysis-dir "$ANALYSIS_DIR" \
  --output-root "$OUTPUT_ROOT/data" \
  "$@" \
  2>&1 | tee "$OUTPUT_ROOT/audit/prepare.log"
