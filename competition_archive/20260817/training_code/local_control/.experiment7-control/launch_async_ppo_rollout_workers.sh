#!/usr/bin/env bash
set -Eeuo pipefail

declare -A addresses=(
  [doraemon02]=10.113.13.53
  [doraemon03]=10.113.13.54
  [doraemon04]=10.113.13.57
  [doraemon08]=10.113.13.63
  [doraemon09]=10.113.13.64
  [doraemon10]=10.113.13.67
  [doraemon11]=10.113.13.68
  [doraemon12]=10.113.13.69
  [doraemon13]=10.113.13.71
  [doraemon14]=10.113.13.72
  [doraemon15]=10.113.13.73
  [doraemon16]=10.113.13.74
  [doraemon17]=10.113.13.75
  [doraemon19]=10.113.13.77
  [doraemon20]=10.113.13.78
)

pids=()
for host in "${!addresses[@]}"; do
  log="/tmp/launch-${host}-async-ppo.log"
  ssh -tt \
    -o BatchMode=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=8 \
    -o ServerAliveInterval=5 \
    -o ServerAliveCountMax=2 \
    "lzhang@${addresses[$host]}" \
    "/bin/bash --noprofile --norc /homes/lzhang/start_async_ppo_rollout_worker.sh $host" \
    >"$log" 2>&1 </dev/null &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=$((failed + 1))
  fi
done
for host in "${!addresses[@]}"; do
  printf '[%s]\n' "$host"
  tail -5 "/tmp/launch-${host}-async-ppo.log" || true
done
echo "LAUNCH_COMPLETE requested=${#addresses[@]} failed=$failed"
