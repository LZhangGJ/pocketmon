#!/usr/bin/env bash
set -u
for host in 10.113.13.63 10.113.13.72 10.113.13.74 10.113.13.75 10.113.13.77 10.113.13.78; do
  echo "HOST=$host"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "bash --noprofile --norc -c 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true'" || true
done
