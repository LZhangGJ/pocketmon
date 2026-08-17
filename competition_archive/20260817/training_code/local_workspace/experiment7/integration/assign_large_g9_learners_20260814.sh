#!/usr/bin/env bash
set -u

runner=/homes/lzhang/transition_ppo_learner_20260814.sh
run() {
  local host=$1 mode=$2 chain=$3 device=$4
  echo "=== $host $mode $chain $device"
  ssh -o BatchMode=yes -o ConnectTimeout=3 "lzhang@$host" \
    "bash $runner $mode $chain $device" 2>&1 || true
}

run 10.113.13.72 stop a02_grim_large_g9 cuda:2 &
run 10.113.13.72 restart lucario_gold_exact cuda:1 &
run 10.113.13.72 restart dragapult_munkidori_large_g9 cuda:0 &
run 10.113.13.73 restart a02_grim_large_g9_pokegear cuda:0 &
run 10.113.13.73 restart a08_maxbelt_large_g9 cuda:1 &
run 10.113.13.54 start alakazam_large_g9 cuda:0 &
run 10.113.13.54 start kangaskhan_crustle_large_g9 cuda:1 &
run 10.113.13.68 start festival_grass_large_g9 cuda:0 &
run 10.113.13.71 start universal_ppo_large_256x6 cuda:0 &
wait
