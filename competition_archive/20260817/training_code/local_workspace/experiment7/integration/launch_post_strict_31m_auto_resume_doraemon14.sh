#!/usr/bin/env bash
set -euo pipefail

profile=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-scaled-20260815/2026-08-13/ultra_512x8_31m
control="$profile/post-strict-31m-controller"
runtime=/dev/shm/lzhang-ultra-bc-runtime-20260815
python=/usr/bin/python3
checkpoint=/dev/shm/lzhang-ultra-bc-scratch-20260815/migration-d14-20260815T0915Z/training/checkpoints/epoch_000002.pt
expected_sha=4a5f332e9c8da2a001373e24c893bbc39c6586ad8fb880df46deab18fd89e4e7
strict_root=/suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/windows/2026-08-14-scoregt1000-20260815T1200Z
monitor=/suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/monitoring/strict-scoregt1000-20260815T1200Z

mkdir -p "$control"
actual_sha=$(sha256sum "$checkpoint" | awk '{print $1}')
test "$actual_sha" = "$expected_sha"

if test -s "$control/controller.pid"; then
  old_pid=$(cat "$control/controller.pid")
  if kill -0 "$old_pid" 2>/dev/null; then
    old_command=$(tr '\0' ' ' <"/proc/$old_pid/cmdline")
    case "$old_command" in
      *post_strict_31m_auto_resume.py*)
        echo "31M_POST_STRICT_CONTROLLER_ALREADY_RUNNING=$old_pid"
        exit 0
        ;;
      *)
        echo "PIDFILE_COMMAND_MISMATCH=$old_pid" >&2
        exit 4
        ;;
    esac
  fi
fi

nohup env \
  PYTHONPATH=/dev/shm/lzhang-ultra-bc-pydeps \
  PYTHONNOUSERSITE=1 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  "$python" -s "$runtime/post_strict_31m_auto_resume.py" \
  --profile-root "$profile" \
  --epoch2-output /dev/shm/lzhang-ultra-bc-scratch-20260815/migration-d14-20260815T0915Z/training \
  --epoch2-checkpoint "$checkpoint" \
  --epoch2-result /dev/shm/lzhang-ultra-bc-scratch-20260815/migration-d14-20260815T0915Z/training/validation-results/epoch_000002.json \
  --epoch2-sha256 "$expected_sha" \
  --strict-success "$strict_root/SUCCESS" \
  --parity-receipt "$monitor/tensor-parity.json" \
  --strict-manifest "$strict_root/tensordict-sources.json" \
  --stage-root /dev/shm/lzhang-ultra-bc-strict-scoregt1000-20260815T1200Z \
  --scratch-root /dev/shm/lzhang-ultra-bc-strict-scratch-20260815 \
  --runtime "$runtime" \
  --python "$python" \
  --output-root /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-scaled-scoregt1000-incremental-20260815/2026-08-14-scoregt1000-20260815T1200Z \
  --baseline-report /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/0812-d14-ram-npz-fast-20260813/large_256x6/async-validation-report.json \
  --attempt-id post-strict-31m-20260815T1318Z \
  --old-controller-pid 2284902 \
  --old-trainer-pid 2284913 \
  --old-validator-pid 2284914 \
  >"$control/controller.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$control/launcher.pid"
sleep 3
kill -0 "$pid"
printf 'STARTED=%s\n' "$pid"
sed -n '1,200p' "$control/state.json"
