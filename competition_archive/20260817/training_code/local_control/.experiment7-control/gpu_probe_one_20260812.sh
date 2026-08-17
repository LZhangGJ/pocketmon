#!/usr/bin/env bash
set -Eeuo pipefail

host_label="${1:-unknown}"
echo "HOST=${host_label}"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA_SMI=missing"
  exit 0
fi

nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader,nounits | sed 's/^/GPU|/'

nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader,nounits 2>/dev/null |
while IFS=, read -r uuid pid process memory; do
  pid="$(printf '%s' "$pid" | xargs)"
  args="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  printf 'PROC|%s|%s|%s|%s|%s\n' \
    "$(printf '%s' "$uuid" | xargs)" "$pid" \
    "$(printf '%s' "$process" | xargs)" \
    "$(printf '%s' "$memory" | xargs)" "$args"
done
