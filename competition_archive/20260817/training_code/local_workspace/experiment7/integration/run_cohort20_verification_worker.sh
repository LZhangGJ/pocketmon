#!/usr/bin/env bash
set -euo pipefail

main_root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
worktree=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
python=/homes/lzhang/mypath/new/envs/trans/bin/python
worker_id="$(hostname)-cohort20-verify"
summary_dir="$main_root/buffer/ready/universal_ppo_large_256x6"
receipt_dir="$main_root/control/large-g9-pool-reconfiguration-20260814"
log="$receipt_dir/$worker_id.log"
pidfile="$receipt_dir/$worker_id.pid"

if [[ -s "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
  printf 'ALREADY_RUNNING=%s\n' "$(cat "$pidfile")"
  exit 0
fi
before=$(find "$summary_dir" -maxdepth 1 -type f -name "$worker_id-*.summary.json" | wc -l)
nohup env PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ionice -c 2 -n 7 nice -n 10 \
  "$python" -s "$worktree/experiment7/integration/run_async_ppo_rollout_worker.py" \
    --league "$main_root/state/league.json" \
    --worktree "$worktree" \
    --buffer-root "$main_root/buffer" \
    --python "$python" \
    --worker-id "$worker_id" \
    --episodes-per-shard 20 \
    --refresh-rounds 1 \
    --self-play-fraction 0.15 \
    --cpu-limit 70 \
    --io-limit 80 \
    --only-chain universal_ppo_large_256x6 \
    >>"$log" 2>&1 </dev/null &
parent=$!
printf '%s\n' "$parent" >"$pidfile"
printf 'STARTED=%s\n' "$parent"

(
  deadline=$((SECONDS + 1800))
  while (( SECONDS < deadline )); do
    current=$(find "$summary_dir" -maxdepth 1 -type f -name "$worker_id-*.summary.json" | wc -l)
    if (( current > before )); then
      kill -STOP "$parent" 2>/dev/null || true
      while pgrep -P "$parent" >/dev/null 2>&1; do sleep 2; done
      kill -TERM "$parent" 2>/dev/null || true
      kill -CONT "$parent" 2>/dev/null || true
      newest=$(ls -1t "$summary_dir"/$worker_id-*.summary.json | head -1)
      printf 'completedAt=%s\nsummary=%s\n' "$(date -u +%FT%TZ)" "$newest" \
        >"$receipt_dir/$worker_id.receipt.txt"
      exit 0
    fi
    sleep 2
  done
  kill -TERM "$parent" 2>/dev/null || true
  printf 'timeoutAt=%s\n' "$(date -u +%FT%TZ)" >"$receipt_dir/$worker_id.failed.txt"
) >>"$log" 2>&1 &
