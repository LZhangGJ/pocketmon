#!/usr/bin/env bash
set -Eeuo pipefail

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
worktree=/homes/lzhang/worktrees/experiment7-async-4c45f89
python=/homes/lzhang/mypath/new/envs/trans/bin/python

handover() {
  local chain="$1" old_pid="$2" min_decisions="$3" teacher_anchor="$4" seat1_weight="$5" log="$6"
  local command_line children new_pid
  command_line=$(tr '\0' ' ' <"/proc/$old_pid/cmdline")
  case "$command_line" in
    *"$worktree/experiment7/integration/run_async_ppo_learner.py"*"--chain $chain"*) ;;
    *) echo "REFUSE_UNEXPECTED_LEARNER chain=$chain pid=$old_pid command=$command_line" >&2; return 2 ;;
  esac

  while kill -0 "$old_pid" 2>/dev/null; do
    children=$(pgrep -P "$old_pid" || true)
    if [[ -z "$children" ]]; then
      kill -STOP "$old_pid"
      children=$(pgrep -P "$old_pid" || true)
      if [[ -z "$children" ]]; then
        kill -TERM "$old_pid"
        kill -CONT "$old_pid" 2>/dev/null || true
        for _ in {1..50}; do
          kill -0 "$old_pid" 2>/dev/null || break
          sleep 0.1
        done
        if kill -0 "$old_pid" 2>/dev/null; then
          echo "REFUSE_FORCE_KILL chain=$chain pid=$old_pid" >&2
          return 3
        fi
        break
      fi
      kill -CONT "$old_pid"
    fi
    sleep 0.05
  done

  nohup env \
    PYTHONNOUSERSITE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    "$python" "$worktree/experiment7/integration/run_async_ppo_learner.py" \
      --league "$root/state/league.json" \
      --chain "$chain" \
      --worktree "$worktree" \
      --run-root "$root/learners" \
      --buffer-root "$root/buffer" \
      --python "$python" \
      --device cuda:0 \
      --max-behavior-lag 2 \
      --min-decisions "$min_decisions" \
      --teacher-anchor-coefficient "$teacher_anchor" \
      --seat1-weight "$seat1_weight" \
      --poll-seconds 10 \
      >"$root/logs/$log" 2>&1 </dev/null &
  new_pid=$!
  sleep 1
  kill -0 "$new_pid"
  printf '%s\n' "$new_pid" >"$root/workers/learner-$chain.tuned.pid"
  printf 'chain=%s old_pid=%s new_pid=%s min_decisions=%s teacher_anchor=%s seat1_weight=%s started_at=%s\n' \
    "$chain" "$old_pid" "$new_pid" "$min_decisions" "$teacher_anchor" "$seat1_weight" "$(date -Iseconds)" \
    >"$root/state/tuned-$chain-handover.txt"
  echo "TUNED_LEARNER_STARTED chain=$chain old_pid=$old_pid new_pid=$new_pid min_decisions=$min_decisions teacher_anchor=$teacher_anchor seat1_weight=$seat1_weight"
}

handover mega_lucario_ex 1879227 6000 0.04 1.0 learner-lucario-tuned.log
handover a08_dipplin_seaking 1879240 4000 0.02 2.0 learner-a08-seat1-tuned.log
