#!/usr/bin/env bash
set -Eeuo pipefail

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
control="$root/control"
pairs=(
  10.113.13.53:doraemon02 10.113.13.54:doraemon03 10.113.13.57:doraemon04
  10.113.13.64:doraemon09 10.113.13.67:doraemon10 10.113.13.68:doraemon11
  10.113.13.69:doraemon12 10.113.13.71:doraemon13 10.113.13.72:doraemon14
  10.113.13.73:doraemon15 10.113.13.74:doraemon16 10.113.13.75:doraemon17
)

for pair in "${pairs[@]}"; do
  host=${pair%%:*}
  worker=${pair##*:}
  remote="nohup bash $control/restart_adaptive_rollout_worker_20260812.sh $worker >$root/logs/restart-adaptive-$worker.log 2>&1 </dev/null &"
  if timeout 8 ssh -o BatchMode=yes "lzhang@$host" "bash --noprofile --norc -c '$remote'"; then
    echo "RESTART_SCHEDULED worker=$worker host=$host"
  else
    echo "RESTART_SKIP worker=$worker host=$host"
  fi
done

timeout 8 ssh -o BatchMode=yes lzhang@10.113.13.72 \
  "bash --noprofile --norc -c 'nohup bash $control/restart_adaptive_learners_20260812.sh d14 >$root/logs/restart-adaptive-learners-d14.log 2>&1 </dev/null &'"
timeout 8 ssh -o BatchMode=yes lzhang@10.113.13.73 \
  "bash --noprofile --norc -c 'nohup bash $control/restart_adaptive_learners_20260812.sh d15 >$root/logs/restart-adaptive-learners-d15.log 2>&1 </dev/null &'"
echo LEARNER_RESTARTS_SCHEDULED
