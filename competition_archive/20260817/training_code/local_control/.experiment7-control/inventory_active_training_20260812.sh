#!/usr/bin/env bash
set -u
hosts=(10.113.13.53 10.113.13.54 10.113.13.57 10.113.13.63 10.113.13.64 10.113.13.67 10.113.13.68 10.113.13.69 10.113.13.71 10.113.13.72 10.113.13.73 10.113.13.74 10.113.13.75 10.113.13.77 10.113.13.78)
for host in "${hosts[@]}"; do
  result=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "bash --noprofile --norc -c 'ps -eo pid,etimes,pcpu,pmem,args | grep -E \"[r]un_async_ppo_learner.py|[t]rain_universal_bc.py|[r]un_a08_deck_variant|[t]rain_universal_ppo.py|[r]un_latest_ppo_full_eval.py|[r]un_latest_ppo_submission4_eval.py|[c]ollect_universal_ppo_rollouts.py\" || true'" 2>/dev/null) || continue
  if [ -n "$result" ]; then
    echo "HOST=$host"
    echo "$result"
  fi
done

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
echo LEAGUE
python3 - "$root/state/league.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for name,row in d.get('chains',{}).items():
    print(name,row.get('generation'),row.get('snapshotId'))
PY

echo A08_BRANCHES
branch=/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-deck-branches-20260812
for name in a08_maxbelt a08_lilligant a08_lilligant_maxbelt; do
  count=$(find "$branch/$name" -path '*/generation-*/metrics.json' -type f 2>/dev/null | wc -l)
  echo "$name generations=$count"
done

echo A100_FASTTRACK
find /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/capacity-comparison-a100-b256 -maxdepth 3 -type f -printf '%T@ %s %p\n' 2>/dev/null | sort -nr | head -20
