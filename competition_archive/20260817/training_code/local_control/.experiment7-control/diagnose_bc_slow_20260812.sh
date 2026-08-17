#!/usr/bin/env bash
set -u
for host in 10.113.13.73 10.113.13.75; do
  echo "===== HOST $host ====="
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "bash --noprofile --norc -c '
    echo LOAD; uptime
    echo MEMORY; free -h | head -2
    echo GPU; nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null || true
    echo GPU_PROCS; nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits 2>/dev/null || true
    echo BC_PROCS; ps -eo pid,etimes,pcpu,pmem,nlwp,stat,args | grep \"[t]rain_universal_bc.py\" || true
    echo IO_PRESSURE; cat /proc/pressure/io 2>/dev/null || true
    echo CPU_PRESSURE; cat /proc/pressure/cpu 2>/dev/null || true
  '" || echo UNREACHABLE
done
