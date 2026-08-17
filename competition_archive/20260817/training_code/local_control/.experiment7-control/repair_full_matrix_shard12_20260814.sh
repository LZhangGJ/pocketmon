#!/usr/bin/env bash
set -euo pipefail

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
eval_root="$root/monitoring/full-matrix"
round="$eval_root/rounds/.20260814T040813Z-a02_grim_g247-g000286-a02_grim_g247_pokegear-g000286-a08_maxbelt-g000306-a08_rabsca-g000306-lucario_gold_exact-g000018-universal_ppo_large_256x6-g000009-universal_ppo_standard_1m-g000011.in-progress"
output="$round/raw/results-shard-012.csv"
if [[ -e "$output" ]]; then
  echo "refusing to overwrite existing output: $output" >&2
  exit 3
fi

exec ssh -T -o BatchMode=yes -o ConnectTimeout=20 \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=12 \
  lzhang@10.113.13.54 \
  /homes/lzhang/run_load_guarded_arena_shard.sh \
  /homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0 \
  /homes/lzhang/mypath/new/envs/trans/bin/python3.11 \
  "$round/schedule.csv" "$round/learners.json" "$round/opponents.json" \
  /dataT0/Free/lzhang/pocketmon-runs/experiment7-opponent-pool-20260810/official-engine/cg \
  "$output" 12 15 "$eval_root"
