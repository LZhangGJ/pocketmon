#!/usr/bin/env bash
set -euo pipefail
source=/tmp/run_async_ppo_rollout_worker.a08-only.py
chain=a08_dipplin_seaking
league=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/state/league.json
buffer=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/buffer
python=/homes/lzhang/mypath/new/envs/trans/bin/python
worktree=/homes/lzhang/worktrees/experiment7-async-4c45f89
for spec in \
  10.113.13.63:a08-extra-doraemon08 \
  10.113.13.72:a08-extra-doraemon12 \
  10.113.13.74:a08-extra-doraemon14 \
  10.113.13.77:a08-extra-doraemon17
do
  host=${spec%%:*}; id=${spec#*:}
  echo "DEPLOY host=$host worker=$id"
  scp -q -o BatchMode=yes -o ConnectTimeout=8 "$source" "$host:$worktree/experiment7/integration/run_async_ppo_rollout_worker.py"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "bash --noprofile --norc -c '
    set -eu
    if pgrep -f \"^$python -s $worktree/experiment7/integration/run_async_ppo_rollout_worker.py .*--worker-id $id( |$)\" >/dev/null; then
      echo ALREADY_RUNNING worker=$id
      exit 0
    fi
    mkdir -p $buffer/logs/$id $buffer/pids
    nohup ionice -c2 -n7 nice -n 10 env PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      $python -s $worktree/experiment7/integration/run_async_ppo_rollout_worker.py \
      --league $league --worktree $worktree --buffer-root $buffer --python $python \
      --worker-id $id --episodes-per-shard 20 --refresh-rounds 1 \
      --cpu-limit 95 --io-limit 80 --only-chain $chain \
      >$buffer/logs/$id/worker.log 2>&1 </dev/null &
    pid=\$!
    echo \$pid > $buffer/pids/$id.pid
    sleep 2
    kill -0 \$pid
    echo STARTED worker=$id pid=\$pid
  '"
done
