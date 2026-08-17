#!/usr/bin/env bash
set -u

for host in "$@"; do
  echo "=== HOST=$host"
  timeout 6 ssh -o BatchMode=yes -o ConnectTimeout=2 "lzhang@$host" '
    hostname
    printf "nproc="; nproc
    uptime
    printf "all_chain_workers="
    pgrep -af run_async_ppo_rollout_worker.py | grep -v -- --only-chain | wc -l
    nvidia-smi pmon -c 1 2>/dev/null | grep -v "^#" || true
    ps -eo pid,ppid,state,etimes,cmd |
      grep -e run_async_ppo_learner.py -e train_universal -e validate_universal |
      grep -v grep || true
  ' 2>&1 || true
done
