#!/usr/bin/env bash
set -u
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
for id in a08-extra-doraemon08 a08-extra-doraemon12 a08-extra-doraemon14 a08-extra-doraemon17; do
  echo "WORKER=$id"
  cat "$root/buffer/pids/$id.pid" 2>/dev/null || echo NO_PID
  tail -3 "$root/buffer/logs/$id/worker.log" 2>/dev/null || echo NO_LOG
  find "$root/buffer/ready/a08_dipplin_seaking" -maxdepth 1 -type f -name "$id-*.summary.json" -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1
done
