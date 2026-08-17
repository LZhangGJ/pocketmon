#!/usr/bin/env bash
set -euo pipefail

runtime=/dataT0/Free/lzhang/pocketmon-runs/experiment7-static-deck-bc-10d-20260815/control/runtime
control=/dataT0/Free/lzhang/pocketmon-runs/experiment7-static-deck-bc-10d-20260815/control/post-strict
strict=/tmp/lzhang-strict-scoregt1000-window-20260815T1200Z
parity=/suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/monitoring/strict-scoregt1000-20260815T1200Z/tensor-parity.json
expected_sha=b80edd05263002dbaf830dc4ad0296b7ede435f7a861461a9f28838a72d5010b
python=/homes/lzhang/mypath/new/envs/trans/bin/python

if pgrep -af '[s]tatic_deck_bc_post_strict_controller.py.*--authority-host doraemon17' >/dev/null; then
  pgrep -af '[s]tatic_deck_bc_post_strict_controller.py.*--authority-host doraemon17'
  exit 0
fi
if [[ -f "$control/controller.lock" ]]; then
  mv "$control/controller.lock" "$control/controller.lock.pre-local-authority.$(date -u +%Y%m%dT%H%M%SZ)"
fi
nohup "$python" -s "$runtime/integration/static_deck_bc_post_strict_controller.py" \
  --config "$runtime/config/static_deck_bc_10d_20260815.json" \
  --strict-root "$strict" \
  --control-root "$control" \
  --launch-command "$runtime/integration/launch_static_deck_bc_d17_authority.sh" \
  --authority-host doraemon17 \
  --expected-manifest-sha256 "$expected_sha" \
  --parity-receipt "$parity" \
  --launch-mode authority_build_only \
  --poll-seconds 15 \
  >"$control/controller-local-authority.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$control/controller-local-authority.pid"
printf '%s\n' "$pid"
