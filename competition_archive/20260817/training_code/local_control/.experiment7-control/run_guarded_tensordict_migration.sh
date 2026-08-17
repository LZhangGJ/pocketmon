#!/usr/bin/env bash
set -Eeuo pipefail

worktree=/homes/lzhang/worktrees/experiment7-async-4c45f89
python=/homes/lzhang/mypath/new/envs/trans/bin/python
builder=/homes/lzhang/build_daily_tensordict_window.py
root=/suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc
mkdir -p "$root/control" "$root/logs"

snapshot() {
  local load_value cores cpu io
  load_value=$(cut -d' ' -f1 /proc/loadavg)
  cores=$(nproc --all)
  cpu=$(awk -v load_value="$load_value" -v cores="$cores" 'BEGIN {printf "%.2f",100*load_value/cores}')
  io=$(awk '/^some / {for(i=1;i<=NF;i++) if($i~/^avg10=/){split($i,a,"=");print a[2];exit}}' /proc/pressure/io 2>/dev/null || echo 0)
  printf '%s %s\n' "$cpu" "${io:-0}"
}

run() {
  exec 9>"$root/control/migration.lock"
  if ! flock -n 9; then echo MIGRATION_ALREADY_RUNNING; exit 0; fi
  local cpu io pid paused=0 status
  while true; do
    read -r cpu io < <(snapshot)
    awk -v cpu="$cpu" -v io="$io" 'BEGIN {exit !(cpu<95&&io<80)}' && break
    echo "RESOURCE_WAIT cpu=$cpu io=$io at=$(date -Iseconds)"
    sleep 30
  done
  setsid env PYTHONNOUSERSITE=1 PYTHONPATH="$worktree/experiment7/integration" \
    ionice -c2 -n7 nice -n10 "$python" -s "$builder" --migration-only & pid=$!
  echo "$pid" >"$root/control/migration-child.pid"
  while kill -0 "$pid" 2>/dev/null; do
    read -r cpu io < <(snapshot)
    if awk -v cpu="$cpu" -v io="$io" 'BEGIN {exit !(cpu>=95||io>=80)}'; then
      if [[ $paused -eq 0 ]]; then
        kill -STOP -- "-$pid" 2>/dev/null || true
        paused=1
        echo "RESOURCE_PAUSE pid=$pid cpu=$cpu io=$io at=$(date -Iseconds)"
      fi
    elif [[ $paused -eq 1 ]] && awk -v cpu="$cpu" -v io="$io" 'BEGIN {exit !(cpu<85&&io<70)}'; then
      kill -CONT -- "-$pid" 2>/dev/null || true
      paused=0
      echo "RESOURCE_RESUME pid=$pid cpu=$cpu io=$io at=$(date -Iseconds)"
    fi
    sleep 30
  done
  [[ $paused -eq 0 ]] || kill -CONT -- "-$pid" 2>/dev/null || true
  set +e; wait "$pid"; status=$?; set -e
  printf '{"status":%d,"completedAt":"%s"}\n' "$status" "$(date -Iseconds)" >"$root/control/migration-exit.json"
  return "$status"
}

if [[ "${1:-}" == "--run" ]]; then run; exit; fi
pidfile="$root/control/migration-controller.pid"
if [[ -s "$pidfile" ]]; then
  old=$(<"$pidfile")
  if [[ "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "MIGRATION_CONTROLLER_ALREADY_RUNNING pid=$old"
    exit 0
  fi
fi
nohup /bin/bash "$0" --run >"$root/logs/migration.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
sleep 2
kill -0 "$pid"
echo "MIGRATION_CONTROLLER_STARTED pid=$pid host=$(hostname)"
