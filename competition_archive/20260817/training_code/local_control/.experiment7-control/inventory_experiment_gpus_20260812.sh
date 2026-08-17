#!/usr/bin/env bash
set -Eeuo pipefail

hosts=(
  10.113.13.53 10.113.13.54 10.113.13.57 10.113.13.63 10.113.13.64
  10.113.13.67 10.113.13.68 10.113.13.69 10.113.13.71 10.113.13.72
  10.113.13.73 10.113.13.74 10.113.13.75 10.113.13.77 10.113.13.78
)

shared_probe=/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-deck-branches-20260812/control/gpu_probe_one_20260812.sh

probe() {
  local host="$1"
  timeout 45s ssh -T \
    -o BatchMode=yes \
    -o ConnectTimeout=8 \
    -o ServerAliveInterval=5 \
    -o ServerAliveCountMax=2 \
    "lzhang@$host" \
    "/bin/bash --noprofile --norc '$shared_probe' '$host'" 2>&1
}

pids=()
for host in "${hosts[@]}"; do
  probe "$host" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=$((failed + 1))
  fi
done
echo "INVENTORY_COMPLETE hosts=${#hosts[@]} failed=$failed at=$(date -Iseconds)"
