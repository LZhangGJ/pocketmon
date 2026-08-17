#!/usr/bin/env bash
set -euo pipefail

main_root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
daily_root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc
script="$main_root/control/watch-replay-then-train-bc-from-ppo.py"
log="$main_root/monitoring/gold-acceleration/bc-from-ppo-0813-watcher.log"
pidfile="$main_root/control/bc-from-ppo-0813-watcher.pid"

mapfile -t pids < <(
  pgrep -f '^/homes/lzhang/mypath/new/envs/trans/bin/python -s .*/watch-replay-then-train-bc-from-ppo.py --main-root ' || true
)
if ((${#pids[@]} > 1)); then
  printf 'duplicate watchers: %s\n' "${pids[*]}" >&2
  exit 2
fi
if ((${#pids[@]} == 1)); then
  kill -TERM "${pids[0]}"
  for _ in {1..20}; do
    kill -0 "${pids[0]}" 2>/dev/null || break
    sleep 0.25
  done
  if kill -0 "${pids[0]}" 2>/dev/null; then
    printf 'watcher did not exit after TERM: %s\n' "${pids[0]}" >&2
    exit 3
  fi
fi

nohup /homes/lzhang/mypath/new/envs/trans/bin/python -s "$script" \
  --main-root "$main_root" \
  --daily-root "$daily_root" \
  --window-end 2026-08-13 \
  --poll-seconds 30 \
  >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
sleep 2
kill -0 "$pid"
printf 'WATCHER_RESTARTED pid=%s script=%s\n' "$pid" "$script"
cat "$main_root/control/bc-from-ppo-0813-state.json"
