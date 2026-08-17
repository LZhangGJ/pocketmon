#!/usr/bin/env bash
set -u

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
for host in 10.113.13.53 10.113.13.54 10.113.13.57 10.113.13.68 10.113.13.69 10.113.13.71 10.113.13.72 10.113.13.73; do
  ssh -o BatchMode=yes -o ConnectTimeout=3 "lzhang@$host" \
    "nohup bash /homes/lzhang/restart_rollout_workers_cpu70_20260814.sh $root >$root/control/large-g9-pool-reconfiguration-20260814/worker-restart-$host.log 2>&1 </dev/null &" &
done
wait
echo CPU70_BOUNDARY_RESTARTS_SCHEDULED
