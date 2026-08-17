#!/usr/bin/env bash
set -Eeuo pipefail

league=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
worktree=/homes/lzhang/worktrees/experiment7-async-4c45f89
python=/homes/lzhang/mypath/new/envs/trans/bin/python
chain=a08_dipplin_seaking
pool="$league/control/a08-targeted-opponents.json"
worker_id=${WORKER_ID:-a08-targeted-$(hostname)}
pidfile="$league/workers/$worker_id.pid"
log="$league/logs/worker-$worker_id.log"

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

wait_for_capacity() {
  while true; do
    read -r cpu io < <(resource_snapshot)
    if awk -v cpu="$cpu" -v io="$io" 'BEGIN { exit !(cpu < 95 && io < 80) }'; then
      printf 'LOAD_GUARD_PASS host=%s cpu=%s io=%s at=%s\n' "$(hostname)" "$cpu" "$io" "$(date -Iseconds)"
      return
    fi
    printf 'LOAD_GUARD_WAIT host=%s cpu=%s io=%s at=%s\n' "$(hostname)" "$cpu" "$io" "$(date -Iseconds)"
    sleep 30
  done
}

run_worker() {
  mkdir -p "$league/buffer/ready/$chain" "$league/buffer/logs/$worker_id"
  sequence=0
  while true; do
    wait_for_capacity
    sequence=$((sequence + 1))
    stamp=$(date +%s%N)
    eval "$("$python" -s - "$league/state/league.json" "$chain" <<'PY'
import json
import shlex
import sys
from pathlib import Path

league = json.loads(Path(sys.argv[1]).read_text())
chain = league["chains"][sys.argv[2]]
current = chain["current"]
values = {
    "reference_root": league["referenceRoot"],
    "engine_catalog": league["engineCatalog"],
    "cg_dir": league["cgDir"],
    "checkpoint": current["checkpoint"],
    "teacher": chain["teacher"],
    "deck": chain["deckPath"],
    "generation": current["generation"],
    "snapshot_id": current["snapshotId"],
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"
    shard_id="$worker_id-$(printf '%08d' "$sequence")-$stamp"
    output="$league/buffer/ready/$chain/$shard_id.jsonl.gz"
    shard_log="$league/buffer/logs/$worker_id/$shard_id.log"
    seed=$(( (stamp ^ sequence) & 2147483647 ))
    set +e
    PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      LD_PRELOAD=/homes/lzhang/mypath/new/envs/wsl4mis/lib/libstdc++.so.6 \
      ionice -c 2 -n 7 nice -n 10 \
      "$python" -s "$worktree/experiment7/integration/collect_universal_ppo_rollouts.py" \
        --reference-root "$reference_root" \
        --engine-catalog "$engine_catalog" \
        --checkpoint "$checkpoint" \
        --teacher "$teacher" \
        --deck "$deck" \
        --pool "$pool" \
        --cg-dir "$cg_dir" \
        --episodes 20 \
        --self-play-fraction 0 \
        --temperature 1.0 \
        --max-decisions 5000 \
        --seed "$seed" \
        --run-id "$shard_id" \
        --behavior-generation "$generation" \
        --behavior-snapshot-id "$snapshot_id" \
        --role generalist \
        --device cpu \
        --output "$output" >"$shard_log" 2>&1
    code=$?
    set -e
    printf 'SHARD_DONE id=%s generation=%s exit=%s at=%s\n' "$shard_id" "$generation" "$code" "$(date -Iseconds)"
    if [[ "$code" -ne 0 ]]; then
      tail -80 "$shard_log" || true
      sleep 60
    fi
  done
}

if [[ "${1:-}" == "--run" ]]; then
  run_worker
  exit
fi

mkdir -p "$league/workers" "$league/logs"
if [[ -s "$pidfile" ]]; then
  old_pid=$(<"$pidfile")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "WORKER_ALREADY_RUNNING worker=$worker_id pid=$old_pid"
    exit 0
  fi
fi
nohup env WORKER_ID="$worker_id" /bin/bash "$0" --run >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
sleep 2
kill -0 "$pid"
echo "WORKER_STARTED worker=$worker_id pid=$pid log=$log"
