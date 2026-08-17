#!/usr/bin/env bash
set -euo pipefail

output_root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-scaled-20260815/2026-08-13
profile_root="$output_root/ultra_512x8_31m"
stage_root=/dev/shm/lzhang-ultra-bc-window-20260815/2026-08-13
scratch_root=/dev/shm/lzhang-ultra-bc-scratch-20260815
runtime=/dev/shm/lzhang-ultra-bc-runtime-20260815
python=/usr/bin/python3
export PYTHONPATH=/dev/shm/lzhang-ultra-bc-pydeps
attempt_id=${SCALED_BC_ATTEMPT_ID:-migration-d14-$(date -u +%Y%m%dT%H%M%SZ)}
attempt_root="$profile_root/attempts/$attempt_id"
mkdir -p "$profile_root" "$attempt_root" "$scratch_root"

mapfile -t existing < <(
  pgrep -af '[s]tage_and_run_scaled_universal_bc.py' \
    | grep -F -- "--output-root $output_root" \
    | awk '{print $1}'
)
if ((${#existing[@]})); then
  printf 'ALREADY_RUNNING=%s\n' "${existing[*]}"
  exit 0
fi

nohup "$python" -s \
  "$runtime/stage_and_run_scaled_universal_bc.py" \
  --sources /suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/windows/2026-08-13/tensordict-sources.json \
  --stage-root "$stage_root" \
  --scratch-root "$scratch_root" \
  --runtime "$runtime" \
  --python "$python" \
  --output-root "$output_root" \
  --baseline-report /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/0812-d14-ram-npz-fast-20260813/large_256x6/async-validation-report.json \
  --train-gpu 0 --validation-gpu 2 --copy-workers 4 \
  --lock-name controller-d14-recovery.lock \
  --attempt-id "$attempt_id" --expected-card-vocab 1268 \
  >"$attempt_root/controller.log" 2>&1 &

pid=$!
temporary="$profile_root/.controller.pid.$pid.tmp"
printf '%s\n' "$pid" >"$temporary"
mv -f "$temporary" "$profile_root/controller.pid"
temporary="$attempt_root/.controller.pid.$pid.tmp"
printf '%s\n' "$pid" >"$temporary"
mv -f "$temporary" "$attempt_root/controller.pid"
printf 'STARTED=%s ATTEMPT=%s HOST=doraemon14\n' "$pid" "$attempt_id"
