#!/usr/bin/env bash
set -euo pipefail

bash -n /tmp/run_load_guarded_arena_shard.sh
install -m 0755 /tmp/run_load_guarded_arena_shard.sh /homes/lzhang/run_load_guarded_arena_shard.sh
install -m 0755 /tmp/reserve_d16_test.py /homes/lzhang/reserve_d16_test.py

if ! pgrep -f '^/usr/bin/python3 /homes/lzhang/reserve_d16_test.py$' >/dev/null; then
  nohup /usr/bin/python3 /homes/lzhang/reserve_d16_test.py \
    >/tmp/reserve_d16_test.log 2>&1 &
  echo "RESERVATION_PID=$!"
fi

sleep 2
sed -n '1,120p' \
  /dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/control/server-reservations/doraemon16.json
ps -o pid,ppid,state,etime,args \
  -p 877742,879817,905581,905595 --no-headers || true
