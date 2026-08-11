#!/usr/bin/env bash
set -euo pipefail

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
declare -A addresses=(
  [doraemon02]=10.113.13.53 [doraemon03]=10.113.13.54 [doraemon04]=10.113.13.57
  [doraemon08]=10.113.13.63 [doraemon09]=10.113.13.64 [doraemon10]=10.113.13.67
  [doraemon11]=10.113.13.68 [doraemon12]=10.113.13.69 [doraemon13]=10.113.13.71
  [doraemon14]=10.113.13.72 [doraemon15]=10.113.13.73 [doraemon16]=10.113.13.74
  [doraemon17]=10.113.13.75 [doraemon19]=10.113.13.77 [doraemon20]=10.113.13.78
)

pids=()
for host in "${!addresses[@]}"; do
  pidfile="$root/workers/$host.pid"
  if [[ ! -s "$pidfile" ]]; then
    echo "WORKER_PID_MISSING host=$host"
    continue
  fi
  pid=$(<"$pidfile")
  timeout 30 ssh -T -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
    "lzhang@${addresses[$host]}" \
    "kill -0 $pid 2>/dev/null && renice 10 -p $pid >/dev/null && ionice -c 2 -n 7 -p $pid && echo PRIORITY_OK host=$host pid=$pid" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=$((failed + 1))
  fi
done
echo "PRIORITY_COMPLETE attempted=${#pids[@]} failed=$failed"
