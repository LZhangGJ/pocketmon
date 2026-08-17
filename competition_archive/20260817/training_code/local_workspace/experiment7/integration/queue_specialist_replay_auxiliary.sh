#!/usr/bin/env bash
set -euo pipefail

first_deck=${1:?first deck required}
second_deck=${2:?second deck required}
log_path=${3:?second-stage log required}
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/monitoring/specialist-replay-auxiliary/prepared

for _ in $(seq 1 1440); do
  if [[ -s "$root/$first_deck/universal_training_sources.json" \
        && -s "$root/$first_deck/universal/features.npz" ]]; then
    exec bash /dev/shm/prepare_specialist_replay_auxiliary.sh "$second_deck" \
      >"$log_path" 2>&1
  fi
  sleep 30
done
echo "timed out waiting for complete sources and features for $first_deck" >&2
exit 3
