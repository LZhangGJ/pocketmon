#!/usr/bin/env bash
set -u
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
printf 'TARGETED_LOG\n'
tail -30 "$root/logs/worker-a08-targeted-doraemon17.log" 2>/dev/null || true
branch=/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-deck-branches-20260812
for name in a08_maxbelt a08_lilligant a08_lilligant_maxbelt; do
  printf 'BRANCH_LOG %s\n' "$name"
  tail -25 "$branch/logs/$name.log" 2>/dev/null || true
done
printf 'RELATED_PROCS_D15\n'
ssh -o BatchMode=yes -o ConnectTimeout=8 10.113.13.75 "bash --noprofile --norc -c 'pgrep -af \"experiment7-a08-deck-branches|run_a08_deck|collect_universal_ppo_rollouts\" || true'" || true
printf 'RELATED_PROCS_D16\n'
ssh -o BatchMode=yes -o ConnectTimeout=8 10.113.13.73 "bash --noprofile --norc -c 'pgrep -af \"experiment7-a08-deck-branches|run_a08_deck|collect_universal_ppo_rollouts\" || true'" || true
