#!/usr/bin/env bash
set -u
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/0812-d14-ram-npz-fast-20260813
for profile in standard_1m large_256x6; do
  echo "PROFILE=${profile}"
  progress="${root}/${profile}/progress.jsonl"
  if [[ -f "${progress}" ]]; then
    echo "shards=$(grep -c train_shard "${progress}")"
    tail -n 4 "${progress}"
  fi
  if [[ -f "${root}/${profile}/train.log" ]]; then
    echo "TRAIN_LOG_TAIL=${profile}"
    tail -n 12 "${root}/${profile}/train.log"
  fi
  if [[ -f "${root}/${profile}/controller-state.json" ]]; then
    echo "CONTROLLER_STATE=${profile}"
    cat "${root}/${profile}/controller-state.json"
  fi
  if [[ -f "${root}/${profile}/rescue-validator-state.json" ]]; then
    echo "RESCUE_STATE=${profile}"
    cat "${root}/${profile}/rescue-validator-state.json"
  fi
  if [[ -f "${root}/${profile}/validation-gpu2-rescue.log" ]]; then
    echo "RESCUE_LOG_TAIL=${profile}"
    tail -n 8 "${root}/${profile}/validation-gpu2-rescue.log"
  fi
  find "${root}/${profile}" -maxdepth 3 -type f \( -name '*checkpoint*' -o -name '*.pt' \) -printf '%TY-%Tm-%TdT%TH:%TM:%TS %p %s\n' 2>/dev/null | tail -n 4
done
echo RECENT_FILES
find "${root}" -maxdepth 4 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -n 50
echo GPU
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader
echo PROCS
ps -eo pid,ppid,etimes,args | grep -E 'persistent_async_bc_controller|train_universal_bc' | grep -v grep || true
ps -p 2034166,2034167 -o pid,ppid,etimes,args --no-headers || true
