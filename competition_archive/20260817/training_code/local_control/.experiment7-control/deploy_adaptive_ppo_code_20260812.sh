#!/usr/bin/env bash
set -Eeuo pipefail

stage=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/control/adaptive-code-20260812
hosts=(
  10.113.13.53 10.113.13.54 10.113.13.57 10.113.13.63 10.113.13.64
  10.113.13.67 10.113.13.68 10.113.13.69 10.113.13.71 10.113.13.72
  10.113.13.73 10.113.13.74 10.113.13.75 10.113.13.77 10.113.13.78
)
files=(
  collect_universal_ppo_rollouts.py
  run_async_ppo_rollout_worker.py
  run_async_ppo_learner.py
)

for host in "${hosts[@]}"; do
  if ! timeout 7 ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "lzhang@$host" \
      'bash --noprofile --norc -c "test -d /homes/lzhang/worktrees/experiment7-async-4c45f89/experiment7/integration"'; then
    echo "DEPLOY_SKIP host=$host reason=unreachable_or_missing_worktree"
    continue
  fi
  for file in "${files[@]}"; do
    target="/homes/lzhang/worktrees/experiment7-async-4c45f89/experiment7/integration/$file"
    timeout 10 ssh -o BatchMode=yes "lzhang@$host" \
      "bash --noprofile --norc -c 'test -e $target.pre-adaptive-20260812 || cp -p $target $target.pre-adaptive-20260812'"
    timeout 10 scp -q -o BatchMode=yes "$stage/$file" "lzhang@$host:$target"
  done
  echo "DEPLOY_OK host=$host worktree=4c45f89"
done

# The A02 learner uses the four-PPO worktree on doraemon15.
host=10.113.13.73
target=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0/experiment7/integration/run_async_ppo_learner.py
if timeout 7 ssh -o BatchMode=yes "lzhang@$host" "bash --noprofile --norc -c 'test -e $target'"; then
  timeout 10 ssh -o BatchMode=yes "lzhang@$host" \
    "bash --noprofile --norc -c 'test -e $target.pre-adaptive-20260812 || cp -p $target $target.pre-adaptive-20260812'"
  timeout 10 scp -q -o BatchMode=yes "$stage/run_async_ppo_learner.py" "lzhang@$host:$target"
  echo "DEPLOY_OK host=$host worktree=4ppo learner=A02"
fi
