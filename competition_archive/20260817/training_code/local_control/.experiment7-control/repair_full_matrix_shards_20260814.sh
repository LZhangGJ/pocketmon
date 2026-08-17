#!/usr/bin/env bash
set -euo pipefail

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
eval_root="$root/monitoring/full-matrix"
round="$eval_root/rounds/.20260814T040813Z-a02_grim_g247-g000286-a02_grim_g247_pokegear-g000286-a08_maxbelt-g000306-a08_rabsca-g000306-lucario_gold_exact-g000018-universal_ppo_large_256x6-g000009-universal_ppo_standard_1m-g000011.in-progress"
worktree=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
python=/homes/lzhang/mypath/new/envs/trans/bin/python3.11
cg_dir=/dataT0/Free/lzhang/pocketmon-runs/experiment7-opponent-pool-20260810/official-engine/cg
shards=(1 2 6 7 11 12)
hosts=(10.113.13.54 10.113.13.57 10.113.13.72 10.113.13.73 10.113.13.75 10.113.13.77)

pids=()
for index in "${!shards[@]}"; do
  shard=${shards[$index]}
  host=${hosts[$index]}
  output=$(printf '%s/raw/results-shard-%03d.csv' "$round" "$shard")
  log=$(printf '%s/logs/repair-shard-%03d-host-%s.log' "$round" "$shard" "${host//./-}")
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite existing output: $output" >&2
    exit 3
  fi
  ssh -T -o BatchMode=yes -o ConnectTimeout=20 \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=12 \
    "lzhang@$host" \
    /homes/lzhang/run_load_guarded_arena_shard.sh \
    "$worktree" "$python" "$round/schedule.csv" \
    "$round/learners.json" "$round/opponents.json" "$cg_dir" \
    "$output" "$shard" 15 "$eval_root" \
    >"$log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" -ne 0 ]]; then
  echo REPAIR_FAILED >&2
  exit 1
fi
echo REPAIR_COMPLETE
