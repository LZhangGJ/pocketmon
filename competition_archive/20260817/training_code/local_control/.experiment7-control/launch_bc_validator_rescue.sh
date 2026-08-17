#!/usr/bin/env bash
set -euo pipefail
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/0812-d14-ram-npz-fast-20260813
sources=/tmp/experiment7-bc-d14-local-20260813/sources.json
script=/tmp/rescue_bc_validator_gpu2.py
python=/homes/lzhang/mypath/new/envs/trans/bin/python
for profile in standard_1m large_256x6; do
  output="${root}/${profile}"
  previous="/tmp/experiment7-bc-d14-local-20260813/${profile}"
  pid_file="${output}/rescue-validator-watcher.pid"
  if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "RESCUE_ALREADY_RUNNING ${profile} $(cat "${pid_file}")"
    continue
  fi
  nohup "${python}" -s "${script}" \
    --profile "${profile}" --output-root "${root}" \
    --previous-root "${previous}" --sources "${sources}" --gpu 2 \
    >"${output}/rescue-validator-watcher.log" 2>&1 &
  echo $! >"${pid_file}"
  echo "RESCUE_STARTED ${profile} $!"
done
