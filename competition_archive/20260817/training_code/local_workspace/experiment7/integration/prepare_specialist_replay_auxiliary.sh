#!/usr/bin/env bash
set -euo pipefail

deck_key=${1:?deck key required}
source_root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/current-leaderboard-scoregt1000
output_root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/monitoring/specialist-replay-auxiliary/prepared
worktree=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
python=/homes/lzhang/mypath/new/envs/trans/bin/python
reference_root=/homes/lzhang/pocketmon-worktrees/experiment7-248c61b/experiment7/reference
engine_catalog=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/rebalance/2026-08-06-part-0-of-2-doraemon02/prepared/engine_catalog.json

raw_root="${source_root}/by_deck/${deck_key}"
destination="${output_root}/${deck_key}"
if [[ ! -d "$raw_root" ]]; then
  echo "missing replay source: $raw_root" >&2
  exit 2
fi
if [[ -e "$destination/SUCCESS" ]]; then
  echo "already complete: $destination"
  exit 0
fi
mkdir -p "$destination"

exec nice -n 10 ionice -c 3 "$python" \
  "$worktree/experiment7/integration/prepare_universal_training_data.py" \
  --reference-root "$reference_root" \
  --raw-root "$raw_root" \
  --engine-catalog "$engine_catalog" \
  --output-root "$destination" \
  --python "$python" \
  --policy-source winners \
  --validation-fraction 0.10
