#!/usr/bin/env bash
set -Eeuo pipefail

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/capacity-comparison-a100-ram-prefetch-b256
python=/homes/lzhang/mypath/new/envs/trans/bin/python
controller=/homes/lzhang/continue_a100_bc_20260813.py

mkdir -p "$root/logs"
chmod 755 "$controller"

launch_one() {
  local profile=$1 gpu=$2 log=$3
  if pgrep -f "continue_a100_bc_20260813.py --profile $profile" >/dev/null; then
    echo "ALREADY_RUNNING profile=$profile"
    return
  fi
  PYTHONNOUSERSITE=1 nohup "$python" -s "$controller" \
    --profile "$profile" --gpu "$gpu" --max-epoch 12 \
    >"$root/logs/$log" 2>&1 </dev/null &
  echo "STARTED profile=$profile gpu=$gpu pid=$!"
}

launch_one standard_1m 3 standard_1m-continuation.log
launch_one large_256x6 7 large_256x6-continuation.log
sleep 3
pgrep -af continue_a100_bc_20260813.py
