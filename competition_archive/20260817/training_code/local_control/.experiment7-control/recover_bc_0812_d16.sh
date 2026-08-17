#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT=/suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/tensordict-cache
SOURCE_MANIFEST=/tmp/bc-0812-tensordict-sources-localref.json
FEATURE_ROOT=/tmp/lzhang-bc-0812-features
META_ROOT=/dev/shm/lzhang-bc-0812-local-metadata
LOCAL_MANIFEST=${META_ROOT}/tensordict-sources.json
RUNTIME=/tmp/experiment7-async-bc-runtime-20260813
EXPERIMENT_ROOT=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812
BASELINE_ROOT=${EXPERIMENT_ROOT}/capacity-comparison-a100-ram-prefetch-b256
OUTPUT_ROOT=${EXPERIMENT_ROOT}/0812-tensordict-localcache-20260813
CONTROL_ROOT=${OUTPUT_ROOT}/control

mkdir -p "${FEATURE_ROOT}" "${META_ROOT}" "${CONTROL_ROOT}"

for profile in standard_1m large_256x6; do
  test -f "${BASELINE_ROOT}/${profile}/best_model.pt"
  test -f "${BASELINE_ROOT}/${profile}/training_report.json"
done

# Retire only the known startup-stalled standard process.
if kill -0 704997 2>/dev/null; then
  command_line=$(tr '\0' ' ' </proc/704997/cmdline)
  case "${command_line}" in
    *lzhang-bc-0812-serial-start-20260813/standard_1m*)
      kill 704997
      ;;
    *)
      echo "Refusing to stop unexpected PID 704997: ${command_line}" >&2
      exit 3
      ;;
  esac
fi

for day in 03 04 05 06 07 08 09 10 11 12; do
  date=2026-08-${day}
  source=${SOURCE_ROOT}/${date}
  feature_destination=${FEATURE_ROOT}/${date}
  metadata_destination=${META_ROOT}/${date}
  test -f "${source}/features_tensordict/meta.json"
  mkdir -p "${feature_destination}" "${metadata_destination}"
  rsync -a "${source}/features_tensordict/" "${feature_destination}/features_tensordict/"
  for cache in token_cache sequence_cache identity_cache; do
    rsync -a "${source}/${cache}/" "${metadata_destination}/${cache}/"
  done
done

export SOURCE_MANIFEST FEATURE_ROOT META_ROOT LOCAL_MANIFEST
/homes/lzhang/mypath/new/envs/trans/bin/python -s - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["SOURCE_MANIFEST"])
payload = json.loads(source.read_text(encoding="utf-8"))
feature_root = Path(os.environ["FEATURE_ROOT"])
meta_root = Path(os.environ["META_ROOT"])
for row in payload["datasets"]:
    date = row["name"]
    row["features"] = str(feature_root / date / "features_tensordict")
    row["tokenCache"] = str(meta_root / date / "token_cache")
    row["sequenceCache"] = str(meta_root / date / "sequence_cache")
    row["identityCache"] = str(meta_root / date / "identity_cache")
destination = Path(os.environ["LOCAL_MANIFEST"])
temporary = destination.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
temporary.replace(destination)
PY

nohup env CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 \
  /homes/lzhang/mypath/new/envs/trans/bin/python -s \
  "${RUNTIME}/run_async_bc_profile_controller_20260813.py" \
  --profile standard_1m --train-gpu 1 --validation-gpu 1 --start-epoch 1 \
  --previous-root "${BASELINE_ROOT}/standard_1m" \
  --output-root "${OUTPUT_ROOT}" --sources "${LOCAL_MANIFEST}" \
  --train-batch-size 128 --validation-batch-size 256 \
  >"${CONTROL_ROOT}/standard-controller.log" 2>&1 &
echo $! >"${CONTROL_ROOT}/standard-controller.pid"

nohup env CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 \
  /homes/lzhang/mypath/new/envs/trans/bin/python -s \
  "${RUNTIME}/run_async_bc_profile_controller_20260813.py" \
  --profile large_256x6 --train-gpu 2 --validation-gpu 2 --start-epoch 1 \
  --previous-root "${BASELINE_ROOT}/large_256x6" \
  --output-root "${OUTPUT_ROOT}" --sources "${LOCAL_MANIFEST}" \
  --train-batch-size 128 --validation-batch-size 256 \
  >"${CONTROL_ROOT}/large-controller.log" 2>&1 &
echo $! >"${CONTROL_ROOT}/large-controller.pid"

date -Is >"${CONTROL_ROOT}/launched-at.txt"
