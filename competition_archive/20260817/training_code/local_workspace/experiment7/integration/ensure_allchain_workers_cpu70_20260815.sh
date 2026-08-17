#!/usr/bin/env bash
set -euo pipefail

target=${1:?target worker count}
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
worktree=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
python=/homes/lzhang/mypath/new/envs/trans/bin/python
host=$(hostname)
receipt="$root/control/large-g9-pool-reconfiguration-20260814/worker-scale-$host.jsonl"
mkdir -p "$root/workers" "$root/logs" "$(dirname "$receipt")"

mapfile -t existing < <(
  pgrep -af '/experiment7/integration/run_async_ppo_rollout_worker.py' |
    grep -v -- '--only-chain' | awk '{print $1}' || true
)
count=${#existing[@]}
for index in $(seq "$((count + 1))" "$target"); do
  worker_id="$host-pool70-$(printf '%03d' "$index")"
  log="$root/logs/worker-$worker_id.log"
  nohup env PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    LD_PRELOAD=/homes/lzhang/mypath/new/envs/wsl4mis/lib/libstdc++.so.6 \
    ionice -c 2 -n 7 nice -n 10 \
    "$python" -s "$worktree/experiment7/integration/run_async_ppo_rollout_worker.py" \
      --league "$root/state/league.json" \
      --worktree "$worktree" \
      --buffer-root "$root/buffer" \
      --python "$python" \
      --worker-id "$worker_id" \
      --episodes-per-shard 20 \
      --refresh-rounds 1 \
      --self-play-fraction 0.15 \
      --cpu-limit 70 \
      --io-limit 80 \
      >>"$log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$root/workers/$worker_id.pid"
  printf '{"at":"%s","host":"%s","worker":"%s","pid":%s,"cpuLimit":70,"ioLimit":80}\n' \
    "$(date -u +%FT%TZ)" "$host" "$worker_id" "$pid" >>"$receipt"
done
sleep 1
active=$(pgrep -af '/experiment7/integration/run_async_ppo_rollout_worker.py' | grep -v -- '--only-chain' | wc -l)
echo "WORKERS host=$host before=$count target=$target active=$active cpuLimit=70"
