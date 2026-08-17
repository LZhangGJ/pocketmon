#!/usr/bin/env bash
set -euo pipefail

main_root=${1:?usage: restart_stopped_rollout_workers_immediately_20260814.sh MAIN_ROOT}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
receipt_dir="$main_root/monitoring/replay-diagnosis-training/worker-restarts/$stamp-$(hostname)-immediate"
mkdir -p "$receipt_dir"

pkill -TERM -f '^/bin/bash /homes/lzhang/pocketmon/experiment7/integration/restart_rollout_workers_at_shard_boundary_20260814.sh ' 2>/dev/null || true
sleep 1

mapfile -t parents < <(pgrep -f '/experiment7/integration/run_async_ppo_rollout_worker.py' || true)
for parent in "${parents[@]}"; do
  [[ -r "/proc/$parent/cmdline" ]] || continue
  command_text=$(tr '\0' ' ' < "/proc/$parent/cmdline")
  [[ "$command_text" == *'--only-chain'* ]] && continue
  [[ "$command_text" == *'/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0/'* ]] || continue
  state=$(ps -o state= -p "$parent" 2>/dev/null | tr -d ' ' || true)
  [[ "$state" == T* ]] || continue

  mapfile -d '' -t argv < "/proc/$parent/cmdline"
  worker_id=unknown
  for ((index=0; index + 1 < ${#argv[@]}; index++)); do
    if [[ "${argv[$index]}" == '--worker-id' ]]; then
      worker_id=${argv[$((index + 1))]}
      break
    fi
  done
  printf '%s\n' "$command_text" > "$receipt_dir/$worker_id.command.txt"
  kill -TERM "$parent" 2>/dev/null || true
  kill -CONT "$parent" 2>/dev/null || true
  for _ in $(seq 1 10); do
    kill -0 "$parent" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$parent" 2>/dev/null; then
    printf 'old stopped parent did not exit after TERM: %s\n' "$parent" > "$receipt_dir/$worker_id.failed.txt"
    continue
  fi

  log="$receipt_dir/$worker_id.log"
  nohup "${argv[@]}" >> "$log" 2>&1 < /dev/null &
  new_parent=$!
  printf 'oldPid=%s\nnewPid=%s\nworkerId=%s\nstartedAt=%s\n' \
    "$parent" "$new_parent" "$worker_id" "$(date -u +%FT%TZ)" \
    > "$receipt_dir/$worker_id.receipt.txt"
done

printf '%s\n' "$receipt_dir"
