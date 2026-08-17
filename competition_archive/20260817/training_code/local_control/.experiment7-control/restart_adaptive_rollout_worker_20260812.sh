#!/usr/bin/env bash
set -Eeuo pipefail

worker_id=${1:?worker id required}
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
pidfile="$root/workers/$worker_id.pid"
launcher=/homes/lzhang/start_async_ppo_rollout_worker.sh

if [[ -s "$pidfile" ]]; then
  old_pid=$(<"$pidfile")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    cmd=$(tr '\0' ' ' <"/proc/$old_pid/cmdline")
    [[ "$cmd" == *"run_async_ppo_rollout_worker.py"*"--worker-id $worker_id"* ]] || {
      echo "REFUSE_PID_MISMATCH worker=$worker_id pid=$old_pid cmd=$cmd"
      exit 2
    }
    children=$(pgrep -P "$old_pid" || true)
    kill "$old_pid"
    for _ in $(seq 1 120); do
      alive=0
      for child in $children; do
        kill -0 "$child" 2>/dev/null && alive=1
      done
      [[ $alive -eq 0 ]] && break
      sleep 5
    done
    for child in $children; do
      kill -0 "$child" 2>/dev/null && {
        echo "DEFER_CHILD_STILL_RUNNING worker=$worker_id child=$child"
        exit 3
      }
    done
  fi
fi

/bin/bash "$launcher" "$worker_id"
echo "ADAPTIVE_WORKER_READY worker=$worker_id host=$(hostname)"
