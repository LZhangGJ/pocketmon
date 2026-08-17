#!/usr/bin/env bash
set -euo pipefail

output_root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-scaled-20260815/2026-08-13
profile_root="$output_root/ultra_512x8_31m"
mkdir -p "$profile_root"
attempt_id=${SCALED_BC_ATTEMPT_ID:-recovery-$(date -u +%Y%m%dT%H%M%SZ)}
attempt_root="$profile_root/attempts/$attempt_id"

if [[ -s "$profile_root/controller.pid" ]]; then
  existing_pid=$(<"$profile_root/controller.pid")
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "ALREADY_RUNNING=$existing_pid"
    exit 0
  fi
fi

mkdir -p "$attempt_root"
nohup /homes/lzhang/mypath/new/envs/trans/bin/python -s \
  /tmp/stage_and_run_scaled_universal_bc.py \
  --sources /suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/windows/2026-08-13/tensordict-sources.json \
  --stage-root /tmp/lzhang-ultra-bc-window-20260815/2026-08-13 \
  --runtime /tmp/lzhang-ultra-bc-runtime-20260815 \
  --output-root "$output_root" \
  --baseline-report /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/0812-d14-ram-npz-fast-20260813/large_256x6/async-validation-report.json \
  --train-gpu 0 --validation-gpu 1 --copy-workers 4 \
  --attempt-id "$attempt_id" --expected-card-vocab 1268 \
  >"$attempt_root/controller.log" 2>&1 &

pid=$!
temporary="$profile_root/.controller.pid.$pid.tmp"
printf '%s\n' "$pid" >"$temporary"
mv -f "$temporary" "$profile_root/controller.pid"
temporary="$attempt_root/.controller.pid.$pid.tmp"
printf '%s\n' "$pid" >"$temporary"
mv -f "$temporary" "$attempt_root/controller.pid"
printf 'STARTED=%s ATTEMPT=%s\n' "$pid" "$attempt_id"
