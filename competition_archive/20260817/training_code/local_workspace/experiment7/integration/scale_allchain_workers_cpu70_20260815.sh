#!/usr/bin/env bash
set -u

ensure=/homes/lzhang/ensure_allchain_workers_cpu70_20260815.sh
run() {
  local host=$1 target=$2
  echo "=== $host target=$target"
  ssh -o BatchMode=yes -o ConnectTimeout=3 "lzhang@$host" "bash $ensure $target" 2>&1 || true
}

# Stage 1 deliberately targets about 25% of logical cores.  The per-shard
# guard remains 70%, leaving room for other users and learner bursts.
run 10.113.13.53 4 &
run 10.113.13.54 4 &
run 10.113.13.57 4 &
run 10.113.13.68 3 &
run 10.113.13.69 8 &
run 10.113.13.71 8 &
run 10.113.13.72 20 &
run 10.113.13.73 20 &
run 10.113.13.75 6 &
wait
