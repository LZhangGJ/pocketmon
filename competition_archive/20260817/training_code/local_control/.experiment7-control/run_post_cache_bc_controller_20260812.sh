#!/usr/bin/env bash
set -Eeuo pipefail

league=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
refresh=/dataT0/Free/lzhang/pocketmon-runs/replay-refresh-20260812
output=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812
worktree=/homes/lzhang/worktrees/experiment7-async-4c45f89
current=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/final_7d/sources.json
rehearsal=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-earlier-scoregt900-20260810/daily/2026-08-01/prepared/universal_training_sources.json
initial=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/final_7d/training/seed-20260812-doraemon16-gpu0/best_model.pt
python=/homes/lzhang/mypath/new/envs/trans/bin/python
pidfile="$output/controller.pid"
log="$output/controller.log"

run_controller() {
  mkdir -p "$output"
  while [[ ! -f "$refresh/CACHES_READY" ]]; do
    printf 'WAIT_CACHES at=%s\n' "$(date -Iseconds)"
    sleep 60
  done
  sources="$output/sources-rolling-2026-08-11.json"
  if [[ ! -s "$sources" ]]; then
    "$python" -s "$league/control/build_rolling_universal_sources_20260812.py" \
      --current "$current" \
      --rehearsal-08-01 "$rehearsal" \
      --new-cache-root "$refresh/cache" \
      --output "$sources"
  fi
  # Two PPO learner hosts plus one pre-existing d03 GPU process were observed.
  # Three new BC GPUs keep the experiment-wide concurrent GPU count at <= 6.
  declare -a hosts=(10.113.13.74 10.113.13.75 10.113.13.78)
  declare -a seeds=(20260815 20260816 20260817)
  pids=()
  for index in 0 1 2; do
    host=${hosts[$index]}
    seed=${seeds[$index]}
    seed_output="$output/training/seed-$seed"
    ssh -T -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
      "lzhang@$host" \
      "/bin/bash --noprofile --norc $league/control/run_incremental_universal_bc_seed_nohash.sh $sources $seed_output $initial $worktree $seed 0" \
      >"$output/launch-seed-$seed.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=$((failed + 1))
    fi
  done
  printf 'TRAINING_COMPLETE failed=%s at=%s\n' "$failed" "$(date -Iseconds)"
  if [[ "$failed" -eq 0 ]]; then
    touch "$output/CANDIDATES_READY"
  fi
}

if [[ "${1:-}" == "--run" ]]; then
  run_controller
  exit
fi
mkdir -p "$output"
if [[ -s "$pidfile" ]]; then
  old_pid=$(<"$pidfile")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "BC_CONTROLLER_ALREADY_RUNNING pid=$old_pid"
    exit 0
  fi
fi
nohup /bin/bash "$0" --run >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
sleep 2
kill -0 "$pid"
echo "BC_CONTROLLER_STARTED pid=$pid log=$log"
