#!/usr/bin/env bash
set -euo pipefail

OLD_RAM=/dev/shm/lzhang-bc-capacity-a100-20260812
OLD_MANIFEST=${OLD_RAM}/universal-10d-sources-ram.json
DAY_SOURCE=/dataT0/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/cache/2026-08-12/prepared/universal
DAY_MANIFEST=/dataT0/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/cache/2026-08-12/prepared/universal_training_sources.json
RAM_ROOT=/dev/shm/lzhang-bc-0812-hybrid-npz-20260813
DAY_RAM=${RAM_ROOT}/2026-08-12
LOCAL_MANIFEST=${RAM_ROOT}/universal-10d-sources-ram.json
RUNTIME=/tmp/experiment7-async-bc-runtime-20260813
EXPERIMENT_ROOT=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812
BASELINE_ROOT=${EXPERIMENT_ROOT}/capacity-comparison-a100-ram-prefetch-b256
OUTPUT_ROOT=${EXPERIMENT_ROOT}/0812-a100-hybrid-npz-20260813
CONTROL_ROOT=${OUTPUT_ROOT}/control

mkdir -p "${DAY_RAM}" "${CONTROL_ROOT}"
test -f "${OLD_MANIFEST}"
test -f "${DAY_MANIFEST}"
test -f "${RUNTIME}/run_async_bc_profile_controller_20260813.py"

echo "$(date -Is) staging only the new 2026-08-12 day"
rsync -a "${DAY_SOURCE}/features.npz" "${DAY_RAM}/features.npz"
for cache in token_cache sequence_cache identity_cache; do
  rsync -a "${DAY_SOURCE}/${cache}/" "${DAY_RAM}/${cache}/"
done

export OLD_MANIFEST DAY_MANIFEST DAY_RAM LOCAL_MANIFEST
/homes/lzhang/mypath/new/envs/trans/bin/python -s - <<'PY'
import json
import os
from pathlib import Path

old = json.loads(Path(os.environ["OLD_MANIFEST"]).read_text(encoding="utf-8"))
day = json.loads(Path(os.environ["DAY_MANIFEST"]).read_text(encoding="utf-8"))
rows = [row for row in old["datasets"] if "2026-08-03" <= row["name"] <= "2026-08-11"]
if "datasets" in day:
    new_rows = day["datasets"]
    if len(new_rows) != 1:
        raise RuntimeError(f"unexpected daily manifest row count: {len(new_rows)}")
    row = new_rows[0]
else:
    row = day["dataset"]
row["name"] = "2026-08-12"
base = Path(os.environ["DAY_RAM"])
row["features"] = str(base / "features.npz")
row["tokenCache"] = str(base / "token_cache")
row["sequenceCache"] = str(base / "sequence_cache")
row["identityCache"] = str(base / "identity_cache")
rows.append(row)
if [row["name"] for row in rows] != [f"2026-08-{day:02d}" for day in range(3, 13)]:
    raise RuntimeError("hybrid window is not exactly 2026-08-03 through 2026-08-12")
old["datasets"] = rows
old["capacityComparison"]["dates"] = [row["name"] for row in rows]
old["storage"] = {
    "kind": "hybrid_ram_npz",
    "reusedDates": [row["name"] for row in rows[:-1]],
    "incrementalDate": rows[-1]["name"],
    "reason": "fast A100 recovery while TensorDict migration remains available for future windows",
}
destination = Path(os.environ["LOCAL_MANIFEST"])
temporary = destination.with_suffix(".tmp")
temporary.write_text(json.dumps(old, separators=(",", ":")) + "\n", encoding="utf-8")
temporary.replace(destination)
PY

launch_profile() {
  local profile=$1
  local train_gpu=$2
  test -f "${BASELINE_ROOT}/${profile}/best_model.pt"
  nohup env PYTHONNOUSERSITE=1 \
    /homes/lzhang/mypath/new/envs/trans/bin/python -s \
    "${RUNTIME}/run_async_bc_profile_controller_20260813.py" \
    --profile "${profile}" --train-gpu "${train_gpu}" --validation-gpu 7 --start-epoch 1 \
    --previous-root "${BASELINE_ROOT}/${profile}" \
    --output-root "${OUTPUT_ROOT}" --sources "${LOCAL_MANIFEST}" \
    >"${CONTROL_ROOT}/${profile}-controller.log" 2>&1 &
  echo $! >"${CONTROL_ROOT}/${profile}-controller.pid"
}

launch_profile standard_1m 1
launch_profile large_256x6 3
date -Is >"${CONTROL_ROOT}/launched-at.txt"
echo "$(date -Is) A100 standard and large controllers launched"
