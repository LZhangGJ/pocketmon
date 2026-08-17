#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
  echo "usage: $0 WORKTREE PYTHON SCHEDULE LEARNERS OPPONENTS CG_DIR OUTPUT SHARD_INDEX SHARD_COUNT WRITABLE_ROOT" >&2
  exit 2
fi

# Percentages below are whole-machine aggregates across every logical CPU.
cpu_limit=${ARENA_CPU_LIMIT_PERCENT:-90}
io_limit=${ARENA_IO_LIMIT_PERCENT:-80}
poll_seconds=${ARENA_GUARD_POLL_SECONDS:-15}

while true; do
  cpu_percent=$(LC_ALL=C top -bn1 | awk '/Cpu\(s\)/ {print 100 - $8; exit}')
  io_percent=$(awk -F'avg10=' '/^some/ {split($2,a," "); print a[1]; exit}' /proc/pressure/io)
  cpu_percent=${cpu_percent:-100}
  io_percent=${io_percent:-100}
  if awk -v cpu="$cpu_percent" -v io="$io_percent" -v cmax="$cpu_limit" -v imax="$io_limit" 'BEGIN {exit ! (cpu < cmax && io < imax)}'; then
    break
  fi
  sleep "$poll_seconds"
done

exec nice -n 10 ionice -c 3 bash /homes/lzhang/run_isolated_arena_shard.sh "$@"
