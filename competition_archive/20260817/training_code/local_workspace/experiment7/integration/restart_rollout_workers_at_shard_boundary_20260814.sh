#!/usr/bin/env bash
set -euo pipefail

main_root=${1:?usage: restart_rollout_workers_at_shard_boundary_20260814.sh MAIN_ROOT}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
receipt_dir="$main_root/monitoring/replay-diagnosis-training/worker-restarts/$stamp-$(hostname)"
mkdir -p "$receipt_dir"

mapfile -t parents < <(
  pgrep -f '/experiment7/integration/run_async_ppo_rollout_worker.py' || true
)

for parent in "${parents[@]}"; do
  [[ -r "/proc/$parent/cmdline" ]] || continue
  command_text=$(tr '\0' ' ' < "/proc/$parent/cmdline")
  [[ "$command_text" == *'--only-chain'* ]] && continue
  [[ "$command_text" == *'/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0/'* ]] || continue

  (
    mapfile -d '' -t argv < "/proc/$parent/cmdline"
    worker_id=unknown
    for ((index=0; index + 1 < ${#argv[@]}; index++)); do
      if [[ "${argv[$index]}" == '--worker-id' ]]; then
        worker_id=${argv[$((index + 1))]}
        break
      fi
    done
    log="$receipt_dir/$worker_id.log"
    printf '%s\n' "$command_text" > "$receipt_dir/$worker_id.command.txt"

    cpu_limit_seen=0
    for ((index=0; index < ${#argv[@]}; index++)); do
      if [[ "${argv[$index]}" == '--cpu-limit' ]]; then
        argv[$((index + 1))]=70
        cpu_limit_seen=1
        break
      fi
    done
    if (( cpu_limit_seen == 0 )); then
      argv+=(--cpu-limit 70)
    fi

    kill -STOP "$parent"
    deadline=$((SECONDS + 900))
    while (( SECONDS < deadline )); do
      active_child=0
      while read -r child; do
        [[ -n "$child" ]] || continue
        state=$(ps -o state= -p "$child" 2>/dev/null | tr -d ' ' || true)
        if [[ -n "$state" && "$state" != Z ]]; then
          active_child=1
          break
        fi
      done < <(pgrep -P "$parent" || true)
      (( active_child == 0 )) && break
      sleep 5
    done

    kill -TERM "$parent" 2>/dev/null || true
    kill -CONT "$parent" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$parent" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$parent" 2>/dev/null; then
      printf 'old parent did not exit after TERM: %s\n' "$parent" > "$receipt_dir/$worker_id.failed.txt"
      exit 1
    fi

    nohup "${argv[@]}" >> "$log" 2>&1 < /dev/null &
    new_parent=$!
    printf 'oldPid=%s\nnewPid=%s\nworkerId=%s\ncpuLimit=70\nstartedAt=%s\n' \
      "$parent" "$new_parent" "$worker_id" "$(date -u +%FT%TZ)" \
      > "$receipt_dir/$worker_id.receipt.txt"
  ) &
done

wait
printf '%s\n' "$receipt_dir"
