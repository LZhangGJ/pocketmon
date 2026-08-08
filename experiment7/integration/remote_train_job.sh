#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 WORKTREE DATASET_MANIFEST RUN_DIR SEED GPU_INDEX" >&2
  exit 2
fi

WORKTREE="$1"
DATASET_MANIFEST="$2"
RUN_DIR="$3"
SEED="$4"
GPU_INDEX="$5"
PYTHON="${PYTHON:-/homes/lzhang/mypath/new/envs/trans/bin/python}"

mkdir -p "$RUN_DIR"
LOCK_FILE="${RUN_DIR}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another process owns $LOCK_FILE" >&2
  exit 3
fi

receipt="$RUN_DIR/job_receipt.json"
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat > "$receipt" <<EOF
{"status":"running","host":"$(hostname)","gpu":$GPU_INDEX,"seed":$SEED,"pid":$$,"started_at":"$started","worktree":"$WORKTREE","dataset_manifest":"$DATASET_MANIFEST"}
EOF

cd "$WORKTREE"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set +e
"$PYTHON" experiment7/integration/train_multideck.py \
  --dataset-manifest "$DATASET_MANIFEST" \
  --output-dir "$RUN_DIR/model" \
  --seed "$SEED" \
  > "$RUN_DIR/train.log" 2>&1
code=$?
set -e
finished="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
status="completed"
if [[ $code -ne 0 ]]; then status="failed"; fi
cat > "$receipt" <<EOF
{"status":"$status","host":"$(hostname)","gpu":$GPU_INDEX,"seed":$SEED,"pid":$$,"started_at":"$started","finished_at":"$finished","exit_code":$code,"worktree":"$WORKTREE","dataset_manifest":"$DATASET_MANIFEST","log":"$RUN_DIR/train.log"}
EOF
exit "$code"
