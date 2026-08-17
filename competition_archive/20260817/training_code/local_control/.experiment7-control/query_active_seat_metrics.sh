#!/usr/bin/env bash
set -u
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/learners
for chain in lucario_gold_exact a02_grim_large_g9 a02_grim_large_g9_pokegear dragapult_munkidori_large_g9 a08_maxbelt_large_g9; do
  echo "====$chain"
  find "$root/$chain" -maxdepth 2 -name metrics.json -type f 2>/dev/null | sort | tail -1
done
