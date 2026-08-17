#!/usr/bin/env bash
set -euo pipefail

PYTHON=/homes/lzhang/mypath/new/envs/trans/bin/python
RUNTIME=/tmp/experiment7-async-bc-runtime-20260813
LOCAL=/tmp/experiment7-bc-d13-local-20260813
SOURCES=${LOCAL}/tensordict-sources.json
EXPERIMENT_ROOT=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812
BASELINE_ROOT=${LOCAL}
OUTPUT_ROOT=${EXPERIMENT_ROOT}/0812-d13-tensordict-fast-20260813
CONTROL=${OUTPUT_ROOT}/control

mkdir -p "${CONTROL}"
test -f "${SOURCES}"
test -f "${RUNTIME}/run_async_bc_profile_controller_20260813.py"

launch() {
  local profile=$1
  local train_gpu=$2
  local validation_gpu=$3
  local log=${CONTROL}/${profile}-controller.log
  nohup env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${PYTHON}" -s "${RUNTIME}/run_async_bc_profile_controller_20260813.py" \
    --profile "${profile}" --train-gpu "${train_gpu}" --validation-gpu "${validation_gpu}" \
    --start-epoch 1 --previous-root "${BASELINE_ROOT}/${profile}" \
    --output-root "${OUTPUT_ROOT}" --sources "${SOURCES}" >"${log}" 2>&1 &
  echo $! >"${CONTROL}/${profile}-controller.pid"
}

launch standard_1m 0 3
launch large_256x6 1 3
date -Is >"${CONTROL}/launched-at.txt"
echo "D13_BC_LAUNCHED"
