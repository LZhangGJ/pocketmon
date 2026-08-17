#!/usr/bin/env bash
set -euo pipefail

source=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/learners/universal_ppo_large_256x6/generation-000051/checkpoint.pt
expected=413d15aac3aca5ebe05b6e5f17e55fb50f7e3a53ad30d49725159dbdd9ab6619
root=/dev/shm/lzhang-static-deck-bc-10d-20260815

test "$(sha256sum "$source" | awk '{print $1}')" = "$expected"
plan=$(cat <<'EOF'
doraemon03 dragapult_munkidori
doraemon03 alakazam_dudunsparce
doraemon15 mega_lopunny_mega_froslass
doraemon15 hydrapple_dipplin_ogerpon
doraemon15 crustle_mega_kangaskhan
doraemon15 ogerpon_only
doraemon12 mega_lucario_hariyama
doraemon12 festival_dipplin
doraemon12 slowking_mega_kangaskhan
doraemon04 raging_bolt_ogerpon_kangaskhan
EOF
)

while read -r host profile; do
  ssh -n "$host" "mkdir -p '$root/$profile'"
  rsync -a "$source" "$host:$root/$profile/initializer.pt" </dev/null
  actual=$(ssh -n "$host" "sha256sum '$root/$profile/initializer.pt'" | awk '{print $1}')
  test "$actual" = "$expected"
  printf '%s\t%s\t%s\n' "$host" "$profile" "$actual"
done <<<"$plan"
