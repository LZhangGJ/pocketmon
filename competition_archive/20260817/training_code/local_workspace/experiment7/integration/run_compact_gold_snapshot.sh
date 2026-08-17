#!/usr/bin/env bash
set -euo pipefail

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
monitor="$root/monitoring/gold-acceleration"
mkdir -p "$monitor"
if [[ -s "$monitor/latest-compact.json" ]]; then
  cp -f "$monitor/latest-compact.json" "$monitor/previous-compact.json"
fi
python3 /homes/lzhang/summarize_async_ppo_league.py \
  | python3 /dev/shm/summarize_compact.py \
      --league-root "$root" \
      --output "$monitor/current-compact.json"
cp -f "$monitor/current-compact.json" "$monitor/latest-compact.json"
python3 -m json.tool "$monitor/current-compact.json"
