#!/usr/bin/env bash
set -u
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812
echo FILES
find "$root" -maxdepth 4 -type f \( -name '*.log' -o -name 'training_report.json' -o -name 'epoch_*.pt' \) -printf '%T@ %s %p\n' 2>/dev/null | sort -nr | head -30
echo SEED16_TREE
find "$root/training/seed-20260816" -maxdepth 3 -type f -printf '%T@ %s %p\n' 2>/dev/null | sort -nr | head -20
echo PROCESS_ENV
pid=$(pgrep -f '[t]rain_universal_bc.py.*seed-20260816' | head -1 || true)
if [ -n "$pid" ]; then
  echo PID=$pid
  tr '\0' '\n' < "/proc/$pid/environ" | grep -E '^(CUDA_VISIBLE_DEVICES|OMP_NUM_THREADS|MKL_NUM_THREADS)=' || true
fi
