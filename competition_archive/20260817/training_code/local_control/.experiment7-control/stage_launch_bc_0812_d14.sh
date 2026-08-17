#!/usr/bin/env bash
set -euo pipefail

RAM=/dev/shm/lzhang-bc-0812-npz-20260813
LOCAL=/tmp/experiment7-bc-d14-local-20260813
RUNTIME=/tmp/experiment7-async-bc-runtime-20260813
FORMAL=/suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/windows/2026-08-12/tensordict-sources.json
EXPERIMENT=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812
SHARED_BASE=${EXPERIMENT}/capacity-comparison-a100-ram-prefetch-b256
OUTPUT=${EXPERIMENT}/0812-d14-ram-npz-fast-20260813
CONTROL=${OUTPUT}/control

mkdir -p "${RAM}" "${LOCAL}/reference" "${LOCAL}/standard_1m" "${LOCAL}/large_256x6" "${CONTROL}"
test -f "${FORMAL}"
test -f "${RUNTIME}/run_async_bc_profile_controller_20260813.py"

io_avg10() { awk -F'[ =]' '/^some/ {print $3}' /proc/pressure/io; }
wait_for_io() {
  # User explicitly authorized a temporary I/O-limit bypass for the one-time
  # RAM preload.  Normal training/Arena/rollout guards remain unchanged.
  return 0
}

source_for() {
  case "$1" in
    2026-08-03|2026-08-04|2026-08-05|2026-08-06|2026-08-07|2026-08-08)
      printf '/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/daily/%s/prepared/universal' "$1" ;;
    2026-08-09|2026-08-10|2026-08-11)
      printf '/dataT0/Free/lzhang/pocketmon-runs/replay-refresh-20260812/cache/%s/prepared/universal' "$1" ;;
    2026-08-12)
      printf '/dataT0/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/cache/%s/prepared/universal' "$1" ;;
    *) return 2 ;;
  esac
}

stage_day() {
  local date=$1 source destination cache
  source=$(source_for "${date}")
  destination=${RAM}/${date}
  test -f "${source}/features.npz"
  mkdir -p "${destination}"
  wait_for_io
  rsync -a "${source}/features.npz" "${destination}/features.npz"
  for cache in token_cache sequence_cache identity_cache; do
    wait_for_io
    rsync -a "${source}/${cache}/" "${destination}/${cache}/"
  done
  echo "$(date -Is) STAGED ${date}"
}

pids=()
for day in 03 04 05 06 07 08 09 10 11 12; do
  stage_day "2026-08-${day}" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
(( failed == 0 )) || { echo STAGING_FAILED >&2; exit 7; }

rsync -a /homes/lzhang/worktrees/experiment7-async-4c45f89/experiment7/reference/ "${LOCAL}/reference/"
for profile in standard_1m large_256x6; do
  rsync -a "${SHARED_BASE}/${profile}/best_model.pt" "${LOCAL}/${profile}/best_model.pt"
  rsync -a "${SHARED_BASE}/${profile}/training_report.json" "${LOCAL}/${profile}/training_report.json"
done

export FORMAL RAM LOCAL
/usr/bin/python3 - <<'PY'
import json
import os
from pathlib import Path
p = json.loads(Path(os.environ["FORMAL"]).read_text(encoding="utf-8"))
p["referenceRoot"] = str(Path(os.environ["LOCAL"]) / "reference")
ram = Path(os.environ["RAM"])
for row in p["datasets"]:
    base = ram / row["name"]
    row["features"] = str(base / "features.npz")
    row["tokenCache"] = str(base / "token_cache")
    row["sequenceCache"] = str(base / "sequence_cache")
    row["identityCache"] = str(base / "identity_cache")
p["storage"] = {"kind": "ram_npz", "host": "doraemon14", "hashVerificationRequired": False}
dst = Path(os.environ["LOCAL"]) / "sources.json"
dst.write_text(json.dumps(p, separators=(",", ":")) + "\n", encoding="utf-8")
PY

launch() {
  local profile=$1 train_gpu=$2
  nohup env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /homes/lzhang/mypath/new/envs/trans/bin/python -s \
    "${RUNTIME}/run_async_bc_profile_controller_20260813.py" \
    --profile "${profile}" --train-gpu "${train_gpu}" --validation-gpu 3 --start-epoch 1 \
    --previous-root "${LOCAL}/${profile}" --output-root "${OUTPUT}" \
    --sources "${LOCAL}/sources.json" >"${CONTROL}/${profile}-controller.log" 2>&1 &
  echo $! >"${CONTROL}/${profile}-controller.pid"
}
launch standard_1m 0
launch large_256x6 1
date -Is >"${CONTROL}/launched-at.txt"
echo D14_RAM_BC_LAUNCHED
