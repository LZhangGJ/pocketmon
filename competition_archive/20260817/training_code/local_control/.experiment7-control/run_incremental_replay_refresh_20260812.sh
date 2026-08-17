#!/usr/bin/env bash
set -Eeuo pipefail

worktree=/homes/lzhang/worktrees/experiment7-async-4c45f89
python=/homes/lzhang/mypath/new/envs/trans/bin/python
reference_root="$worktree/experiment7/reference"
engine_catalog=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/rebalance/2026-08-06-part-0-of-2-doraemon02/prepared/engine_catalog.json
root=/dataT0/Free/lzhang/pocketmon-runs/replay-refresh-20260812
raw_root=/dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays/_staging/refresh-20260812
pidfile="$root/controller.pid"
log="$root/controller.log"

resource_snapshot() {
  local cores load_value cpu io
  cores=$(nproc --all)
  load_value=$(awk '{print $1}' /proc/loadavg)
  cpu=$(awk -v load_value="$load_value" -v cores="$cores" 'BEGIN { printf "%.2f", 100.0 * load_value / cores }')
  io=0
  if [[ -r /proc/pressure/io ]]; then
    io=$(awk '/^some / {for(i=1;i<=NF;i++) if($i ~ /^avg10=/){split($i,a,"="); print a[2]; exit}}' /proc/pressure/io)
  fi
  printf '%s %s\n' "$cpu" "${io:-0}"
}

run_guarded() {
  local paused=0 pid code cpu io
  while true; do
    read -r cpu io < <(resource_snapshot)
    if awk -v cpu="$cpu" -v io="$io" 'BEGIN { exit !(cpu < 95 && io < 80) }'; then
      break
    fi
    printf 'LOAD_GUARD_WAIT cpu=%s io=%s at=%s\n' "$cpu" "$io" "$(date -Iseconds)"
    sleep 30
  done
  setsid ionice -c 2 -n 7 nice -n 10 "$@" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    read -r cpu io < <(resource_snapshot)
    if awk -v cpu="$cpu" -v io="$io" 'BEGIN { exit !(cpu >= 95 || io >= 80) }'; then
      if [[ "$paused" -eq 0 ]]; then
        kill -STOP -- "-$pid" 2>/dev/null || true
        paused=1
        printf 'LOAD_GUARD_PAUSE pid=%s cpu=%s io=%s at=%s\n' "$pid" "$cpu" "$io" "$(date -Iseconds)"
      fi
    elif [[ "$paused" -eq 1 ]] && awk -v cpu="$cpu" -v io="$io" 'BEGIN { exit !(cpu < 60 && io < 60) }'; then
      kill -CONT -- "-$pid" 2>/dev/null || true
      paused=0
      printf 'LOAD_GUARD_RESUME pid=%s cpu=%s io=%s at=%s\n' "$pid" "$cpu" "$io" "$(date -Iseconds)"
    fi
    sleep 30
  done
  if [[ "$paused" -eq 1 ]]; then
    kill -CONT -- "-$pid" 2>/dev/null || true
  fi
  set +e
  wait "$pid"
  code=$?
  set -e
  return "$code"
}

run_controller() {
  mkdir -p "$root/audits" "$root/logs" "$raw_root"
  for date in 2026-08-09 2026-08-10 2026-08-11; do
    audit="$root/audits/$date.json"
    if [[ ! -s "$audit" ]]; then
      echo "DOWNLOAD_START date=$date at=$(date -Iseconds)"
      run_guarded env PYTHONNOUSERSITE=1 "$python" -s "$worktree/scripts/download_ptcg_data.py" \
        --date "$date" \
        --max-episodes 0 \
        --data-dir "$raw_root" \
        --audit-output "$audit" \
        >"$root/logs/download-$date.log" 2>&1
      echo "DOWNLOAD_DONE date=$date at=$(date -Iseconds)"
    else
      echo "DOWNLOAD_SKIP_EXISTING_AUDIT date=$date"
    fi

    prepared="$root/cache/$date/prepared"
    if [[ ! -s "$prepared/universal_training_sources.json" ]]; then
      echo "CACHE_START date=$date at=$(date -Iseconds)"
      mkdir -p "$root/cache/$date"
      run_guarded env PYTHONNOUSERSITE=1 "$python" -s "$worktree/experiment7/integration/prepare_universal_training_data.py" \
        --reference-root "$reference_root" \
        --raw-root "$raw_root/$date" \
        --engine-catalog "$engine_catalog" \
        --output-root "$prepared" \
        --python "$python" \
        --policy-source winners \
        --module-versions '*' \
        --validation-fraction 0.05 \
        --strict-catalog \
        --min-game-score-exclusive 900 \
        >"$root/logs/cache-$date.log" 2>&1
      echo "CACHE_DONE date=$date at=$(date -Iseconds)"
    else
      echo "CACHE_SKIP_EXISTING date=$date"
    fi
  done
  touch "$root/CACHES_READY"
  echo "CACHES_READY at=$(date -Iseconds)"
}

if [[ "${1:-}" == "--run" ]]; then
  run_controller
  exit
fi

mkdir -p "$root"
if [[ -s "$pidfile" ]]; then
  old_pid=$(<"$pidfile")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "REFRESH_ALREADY_RUNNING pid=$old_pid"
    exit 0
  fi
fi
nohup /bin/bash "$0" --run >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
sleep 2
kill -0 "$pid"
echo "REFRESH_STARTED pid=$pid log=$log root=$root"
