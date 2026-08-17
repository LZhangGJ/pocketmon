#!/usr/bin/env bash
set -Eeuo pipefail

targets=(
  /dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays/2026-07-25
  /dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays/2026-07-26
  /dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays/2026-07-27
  /dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays/2026-07-28
  /dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays/2026-07-29
  /dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays/2026-07-30
  /dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays/2026-07-31
  /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-earlier-scoregt900-20260810/daily/2026-07-25
  /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-earlier-scoregt900-20260810/daily/2026-07-26
  /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-earlier-scoregt900-20260810/daily/2026-07-27
  /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-earlier-scoregt900-20260810/daily/2026-07-28
  /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-earlier-scoregt900-20260810/daily/2026-07-29
  /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-earlier-scoregt900-20260810/daily/2026-07-30
  /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-earlier-scoregt900-20260810/daily/2026-07-31
)

wait_for_capacity() {
  local cores load_value cpu io
  while true; do
    cores=$(nproc --all)
    load_value=$(awk '{print $1}' /proc/loadavg)
    cpu=$(awk -v load_value="$load_value" -v cores="$cores" 'BEGIN {printf "%.2f",100*load_value/cores}')
    io=$(awk '/^some / {for(i=1;i<=NF;i++)if($i~/^avg10=/){split($i,a,"=");print a[2];exit}}' /proc/pressure/io)
    if awk -v cpu="$cpu" -v io="${io:-0}" 'BEGIN {exit !(cpu<70&&io<70)}'; then return; fi
    echo "CLEANUP_WAIT cpu=$cpu io=${io:-0} at=$(date -Iseconds)"
    sleep 30
  done
}

for target in "${targets[@]}"; do
  resolved=$(realpath -m "$target")
  case "$resolved" in
    /dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays/2026-07-2[5-9]|\
    /dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays/2026-07-3[01]|\
    /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-earlier-scoregt900-20260810/daily/2026-07-2[5-9]|\
    /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-earlier-scoregt900-20260810/daily/2026-07-3[01]) ;;
    *) echo "REFUSE_BAD_TARGET=$resolved" >&2; exit 4 ;;
  esac
  if [[ ! -e "$resolved" ]]; then echo "DELETE_SKIP_MISSING path=$resolved"; continue; fi
  wait_for_capacity
  echo "DELETE_START path=$resolved at=$(date -Iseconds)"
  ionice -c 3 nice -n 19 rm -rf -- "$resolved"
  test ! -e "$resolved"
  echo "DELETE_DONE path=$resolved at=$(date -Iseconds)"
done
echo "CLEANUP_COMPLETE at=$(date -Iseconds)"
