#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 ROLE GENERATION PPO_ROOT OUTPUT_ROOT EXPORT_SCRIPT" >&2
  exit 2
fi

role="$1"
generation="$2"
ppo_root=$(realpath "$3")
output_root="$4"
export_script=$(realpath "$5")
if [[ ! "$generation" =~ ^[0-9]+$ ]]; then
  echo "generation must be numeric: $generation" >&2
  exit 2
fi
if [[ -e "$output_root" ]]; then
  echo "refusing to overwrite output: $output_root" >&2
  exit 3
fi

generation_root=$(printf "%s/%s/generation-%04d" "$ppo_root" "$role" "$generation")
checkpoint="$generation_root/checkpoint.pt"
metrics="$generation_root/metrics.json"
rollouts="$generation_root/rollouts.jsonl.gz"
deadline=$((SECONDS + 21600))
echo "WAIT_START role=$role generation=$generation root=$generation_root"
while [[ ! -s "$checkpoint" || ! -s "$metrics" || ! -s "$rollouts" ]]; do
  if (( SECONDS >= deadline )); then
    echo "WAIT_TIMEOUT role=$role generation=$generation" >&2
    exit 4
  fi
  sleep 30
done
echo "INPUT_COMPLETE role=$role generation=$generation"
exec bash "$export_script" "$role" "$generation" "$ppo_root" "$output_root"
