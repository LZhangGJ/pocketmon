#!/usr/bin/env bash
set -u
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
printf 'FULL_EVAL='
pgrep -fc '[r]un_latest_ppo_full_eval.py' || true
printf 'SUB4_EVAL='
pgrep -fc '[r]un_latest_ppo_submission4_eval.py' || true
printf 'CONTROLLER='
pgrep -fc '[a]daptive_ppo_training_controller.py' || true
python3 - "$root/monitoring/full-matrix/latest.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print('LATEST', d.get('status'), d.get('updatedAt'), d.get('roundId'))
PY
