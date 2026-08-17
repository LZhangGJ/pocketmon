#!/usr/bin/env bash
set -Eeuo pipefail

profile=${1:?usage: start_adjusted_async_ppo_learners.sh d14|d15}
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
python=/homes/lzhang/mypath/new/envs/trans/bin/python

case "$profile" in
  d14)
    worktree=/homes/lzhang/worktrees/experiment7-async-4c45f89
    chains=(a05_raging_bolt_ogerpon_kangaskhan mega_lucario_ex a08_dipplin_seaking)
    ;;
  d15)
    worktree=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
    chains=(a02_submission4_grimmsnarl_froslass_munkidori)
    ;;
  *)
    echo "unknown profile: $profile" >&2
    exit 2
    ;;
esac

start_learner() {
  local chain="$1" min_decisions="$2" teacher_anchor="$3" seat1_weight="$4" log="$5"
  shift 5
  local -a extra=("$@")
  local matches new_pid

  matches=$(pgrep -af "$worktree/experiment7/integration/run_async_ppo_learner.py.*--chain $chain" || true)
  if [[ -n "$matches" ]]; then
    echo "REFUSE_DUPLICATE chain=$chain matches=$matches" >&2
    return 3
  fi

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
      "${extra[@]}" \
      >"$root/logs/$log" 2>&1 </dev/null &
  new_pid=$!
  sleep 2
  kill -0 "$new_pid"
  printf '%s\n' "$new_pid" >"$root/workers/learner-$chain.adjusted.pid"
  printf 'chain=%s new_pid=%s min_decisions=%s teacher_anchor=%s seat1_weight=%s extra=%q started_at=%s\n' \
    "$chain" "$new_pid" "$min_decisions" "$teacher_anchor" "$seat1_weight" "${extra[*]:-}" "$(date -Iseconds)" \
    >"$root/state/adjusted-$chain-start.txt"
  echo "ADJUSTED_LEARNER_STARTED chain=$chain pid=$new_pid min_decisions=$min_decisions teacher_anchor=$teacher_anchor seat1_weight=$seat1_weight extra=${extra[*]:-}"
}

for chain in "${chains[@]}"; do
  case "$chain" in
    a05_raging_bolt_ogerpon_kangaskhan)
      start_learner "$chain" 5000 0.02 1.0 learner-a05-min5000.log
      ;;
    mega_lucario_ex)
      start_learner "$chain" 6000 0.04 1.0 learner-lucario-stable.log
      ;;
    a08_dipplin_seaking)
      start_learner "$chain" 4000 0.02 2.0 learner-a08-seat-balanced.log \
        --normalize-advantages-by-player --balance-player-minibatches
      ;;
    a02_submission4_grimmsnarl_froslass_munkidori)
      start_learner "$chain" 4000 0.02 1.0 learner-a02-min4000.log
      ;;
  esac
done
