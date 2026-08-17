#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=/suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/tensordict-cache
SOURCE_MANIFEST=/suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/windows/2026-08-12/tensordict-sources.json
RAM_ROOT=/dev/shm/lzhang-bc-0812-tensordict-20260813
LOCAL_MANIFEST=${RAM_ROOT}/tensordict-sources.json
RUNTIME=/tmp/experiment7-async-bc-runtime-20260813
EXPERIMENT_ROOT=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812
BASELINE_ROOT=${EXPERIMENT_ROOT}/capacity-comparison-a100-ram-prefetch-b256
OUTPUT_ROOT=${EXPERIMENT_ROOT}/0812-a100-tensordict-20260813
CONTROL_ROOT=${OUTPUT_ROOT}/control

mkdir -p "${RAM_ROOT}" "${CONTROL_ROOT}"
test -f "${SOURCE_MANIFEST}"
test -f "${RUNTIME}/run_async_bc_profile_controller_20260813.py"
for profile in standard_1m large_256x6; do
  test -f "${BASELINE_ROOT}/${profile}/best_model.pt"
  test -f "${BASELINE_ROOT}/${profile}/training_report.json"
done

available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
shm_available_kib=$(df -Pk /dev/shm | awk 'NR==2 {print $4}')
if (( available_kib < 250 * 1024 * 1024 )); then
  echo "Insufficient MemAvailable before RAM staging: ${available_kib} KiB" >&2
  exit 4
fi
if (( shm_available_kib < 100 * 1024 * 1024 )); then
  echo "Insufficient /dev/shm capacity before RAM staging: ${shm_available_kib} KiB" >&2
  exit 5
fi

stage_date() {
  local date=$1
  local source=${SOURCE_ROOT}/${date}
  local destination=${RAM_ROOT}/${date}
  test -f "${source}/features_tensordict/meta.json"
  mkdir -p "${destination}"
  echo "$(date -Is) staging ${date}"
  # One worker per date avoids the pathological single-stream throughput of the
  # shared mount while preserving one sequential reader for each cache tree.
  rsync -a "${source}/features_tensordict/" "${destination}/features_tensordict/"
  for cache in token_cache sequence_cache identity_cache; do
    rsync -a "${source}/${cache}/" "${destination}/${cache}/"
  done
  echo "$(date -Is) staged ${date}"
}

stage_pids=()
for day in 03 04 05 06 07 08 09 10 11 12; do
  stage_date "2026-08-${day}" &
  stage_pids+=("$!")
done
stage_failure=0
for pid in "${stage_pids[@]}"; do
  wait "${pid}" || stage_failure=1
done
if (( stage_failure != 0 )); then
  echo "At least one daily RAM staging worker failed" >&2
  exit 6
fi

export SOURCE_MANIFEST RAM_ROOT LOCAL_MANIFEST
/homes/lzhang/mypath/new/envs/trans/bin/python -s - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["SOURCE_MANIFEST"])
payload = json.loads(source.read_text(encoding="utf-8"))
ram_root = Path(os.environ["RAM_ROOT"])
for row in payload["datasets"]:
    date = row["name"]
    base = ram_root / date
    row["features"] = str(base / "features_tensordict")
    row["tokenCache"] = str(base / "token_cache")
    row["sequenceCache"] = str(base / "sequence_cache")
    row["identityCache"] = str(base / "identity_cache")
destination = Path(os.environ["LOCAL_MANIFEST"])
temporary = destination.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(destination)
PY

nohup env PYTHONNOUSERSITE=1 \
  /homes/lzhang/mypath/new/envs/trans/bin/python -s \
  "${RUNTIME}/run_async_bc_profile_controller_20260813.py" \
  --profile standard_1m --train-gpu 1 --validation-gpu 7 --start-epoch 1 \
  --previous-root "${BASELINE_ROOT}/standard_1m" \
  --output-root "${OUTPUT_ROOT}" --sources "${LOCAL_MANIFEST}" \
  --train-batch-size 256 --validation-batch-size 512 \
  >"${CONTROL_ROOT}/standard-controller.log" 2>&1 &
echo $! >"${CONTROL_ROOT}/standard-controller.pid"

nohup env PYTHONNOUSERSITE=1 \
  /homes/lzhang/mypath/new/envs/trans/bin/python -s \
  "${RUNTIME}/run_async_bc_profile_controller_20260813.py" \
  --profile large_256x6 --train-gpu 3 --validation-gpu 7 --start-epoch 1 \
  --previous-root "${BASELINE_ROOT}/large_256x6" \
  --output-root "${OUTPUT_ROOT}" --sources "${LOCAL_MANIFEST}" \
  --train-batch-size 256 --validation-batch-size 256 \
  >"${CONTROL_ROOT}/large-controller.log" 2>&1 &
echo $! >"${CONTROL_ROOT}/large-controller.pid"

date -Is >"${CONTROL_ROOT}/launched-at.txt"
echo "$(date -Is) A100 BC controllers launched"
