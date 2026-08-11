#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
  echo "usage: $0 WORKTREE PYTHON SCHEDULE LEARNERS OPPONENTS CG_DIR OUTPUT SHARD_INDEX SHARD_COUNT WRITABLE_ROOT" >&2
  exit 2
fi

cpu_limit=${ARENA_CPU_LIMIT_PERCENT:-70}
io_limit=${ARENA_IO_LIMIT_PERCENT:-70}
poll_seconds=${ARENA_LOAD_POLL_SECONDS:-15}

resource_snapshot() {
  local cpu io cores load_value
  cores=$(nproc)
  load_value=$(awk '{print $1}' /proc/loadavg)
  cpu=$(awk -v load_value="$load_value" -v cores="$cores" 'BEGIN { printf "%.2f", 100.0 * load_value / cores }')
  io=0
  if [[ -r /proc/pressure/io ]]; then
    io=$(awk '/^some / {for(i=1;i<=NF;i++) if($i ~ /^avg10=/){split($i,a,"="); print a[2]; exit}}' /proc/pressure/io)
  fi
  printf '%s %s\n' "$cpu" "${io:-0}"
}

while true; do
  read -r cpu_percent io_percent < <(resource_snapshot)
  if awk -v cpu="$cpu_percent" -v io="$io_percent" -v cpu_limit="$cpu_limit" -v io_limit="$io_limit" \
      'BEGIN { exit ! (cpu < cpu_limit && io < io_limit) }'; then
    break
  fi
  echo "ARENA_LOAD_GUARD_WAIT host=$(hostname) cpu=${cpu_percent}% io_pressure=${io_percent}%" >&2
  sleep "$poll_seconds"
done

echo "ARENA_LOAD_GUARD_START host=$(hostname) cpu=${cpu_percent}% io_pressure=${io_percent}%" >&2
exec ionice -c 2 -n 7 nice -n 10 /bin/bash /homes/lzhang/run_isolated_arena_shard.sh "$@"
