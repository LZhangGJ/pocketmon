#!/usr/bin/env bash
set -euo pipefail

runtime=/dev/shm/lzhang-ultra-bc-runtime-20260815
python=/usr/bin/python3
export PYTHONPATH=/dev/shm/lzhang-ultra-bc-pydeps
profile=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-scaled-20260815/2026-08-13/ultra_512x8_31m
strict=/tmp/lzhang-strict-scoregt1000-window-20260815T1200Z
parity=/suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/monitoring/strict-scoregt1000-20260815T1200Z/tensor-parity.json
expected_sha=b80edd05263002dbaf830dc4ad0296b7ede435f7a861461a9f28838a72d5010b
control=$profile/post-strict-31m-controller

if pgrep -af '[p]ost_strict_31m_auto_resume.py.*--strict-success /tmp/lzhang-strict-scoregt1000-window-20260815T1200Z/SUCCESS' >/dev/null; then
  pgrep -af '[p]ost_strict_31m_auto_resume.py.*--strict-success /tmp/lzhang-strict-scoregt1000-window-20260815T1200Z/SUCCESS'
  exit 0
fi
nohup "$python" -s "$runtime/post_strict_31m_auto_resume.py" \
  --profile-root "$profile" \
  --epoch2-output /dev/shm/lzhang-ultra-bc-scratch-20260815/migration-d14-20260815T0915Z/training \
  --epoch2-checkpoint /dev/shm/lzhang-ultra-bc-scratch-20260815/migration-d14-20260815T0915Z/training/checkpoints/epoch_000002.pt \
  --epoch2-result /dev/shm/lzhang-ultra-bc-scratch-20260815/migration-d14-20260815T0915Z/training/validation-results/epoch_000002.json \
  --epoch2-sha256 4a5f332e9c8da2a001373e24c893bbc39c6586ad8fb880df46deab18fd89e4e7 \
  --strict-success "$strict/SUCCESS" \
  --parity-receipt "$parity" \
  --strict-manifest "$strict/tensordict-sources.json" \
  --expected-strict-manifest-sha256 "$expected_sha" \
  --gpu-pair 0 2 \
  --stage-root /dev/shm/lzhang-ultra-bc-strict-scoregt1000-20260815T1200Z \
  --scratch-root /dev/shm/lzhang-ultra-bc-strict-scratch-20260815 \
  --runtime "$runtime" \
  --python "$python" \
  --output-root /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-scaled-scoregt1000-incremental-20260815/2026-08-14-scoregt1000-20260815T1200Z \
  --baseline-report /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/0812-d14-ram-npz-fast-20260813/large_256x6/async-validation-report.json \
  --attempt-id post-strict-31m-local-authority-20260815T1515Z \
  --old-controller-pid 2284902 \
  --old-trainer-pid 2284913 \
  --old-validator-pid 2284914 \
  >"$control/controller-local-authority.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$control/controller-local-authority.pid"
printf '%s\n' "$pid"
