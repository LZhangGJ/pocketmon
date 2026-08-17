#!/usr/bin/env bash
set -eu
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
if pgrep -f '[r]un_latest_ppo_submission4_eval.py' >/dev/null; then
  echo SUB4_EVAL_ALREADY_RUNNING
  exit 0
fi
log="$root/logs/submission4-eval-heartbeat-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup env PYTHONNOUSERSITE=1 /homes/lzhang/mypath/new/envs/trans/bin/python -s /homes/lzhang/run_latest_ppo_submission4_eval.py \
  --league-root "$root" \
  --worktree /homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0 \
  --python /homes/lzhang/mypath/new/envs/trans/bin/python \
  --run-shard /homes/lzhang/run_isolated_arena_shard.sh \
  --summarizer /homes/lzhang/summarize_arena_matrix.py \
  --games-per-agent 20 --shards 16 >"$log" 2>&1 </dev/null &
pid=$!
sleep 2
if kill -0 "$pid" 2>/dev/null; then
  echo "SUB4_EVAL_STARTED pid=$pid log=$log"
else
  echo "SUB4_EVAL_EXITED_EARLY log=$log"
  tail -40 "$log" || true
  exit 1
fi
