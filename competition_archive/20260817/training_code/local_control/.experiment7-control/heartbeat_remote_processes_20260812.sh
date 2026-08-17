#!/usr/bin/env bash
set -u
printf 'TARGETED_D17='
ssh -o BatchMode=yes -o ConnectTimeout=8 10.113.13.77 "bash --noprofile --norc -c 'pgrep -fc \"[r]un_a08_targeted_rollout_worker.sh\" || true'" || echo unreachable
printf 'BRANCH_D15='
ssh -o BatchMode=yes -o ConnectTimeout=8 10.113.13.75 "bash --noprofile --norc -c 'pgrep -af \"[r]un_a08_deck_variant_branch_20260812.sh\" || true'" || echo unreachable
printf 'BRANCH_D16='
ssh -o BatchMode=yes -o ConnectTimeout=8 10.113.13.73 "bash --noprofile --norc -c 'pgrep -af \"[r]un_a08_deck_variant_branch_20260812.sh\" || true'" || echo unreachable
printf 'CAPACITY_D15='
ssh -o BatchMode=yes -o ConnectTimeout=8 10.113.13.75 "bash --noprofile --norc -c 'pgrep -af \"[t]rain_universal_bc.py\" || true'" || echo unreachable
