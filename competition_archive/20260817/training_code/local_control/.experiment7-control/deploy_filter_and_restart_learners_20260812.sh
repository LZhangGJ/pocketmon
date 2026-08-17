#!/usr/bin/env bash
set -Eeuo pipefail
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
stage="$root/control/adaptive-code-20260812"
scp -q "$stage/run_async_ppo_learner.py" \
  lzhang@10.113.13.72:/homes/lzhang/worktrees/experiment7-async-4c45f89/experiment7/integration/run_async_ppo_learner.py
scp -q "$stage/run_async_ppo_learner.py" \
  lzhang@10.113.13.73:/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0/experiment7/integration/run_async_ppo_learner.py
ssh lzhang@10.113.13.72 "bash --noprofile --norc -c 'nohup bash $root/control/restart_adaptive_learners_20260812.sh d14 >$root/logs/restart-adaptive-filter-learners-d14.log 2>&1 </dev/null &'"
ssh lzhang@10.113.13.73 "bash --noprofile --norc -c 'nohup bash $root/control/restart_adaptive_learners_20260812.sh d15 >$root/logs/restart-adaptive-filter-learners-d15.log 2>&1 </dev/null &'"
echo FILTER_RELOAD_SCHEDULED
