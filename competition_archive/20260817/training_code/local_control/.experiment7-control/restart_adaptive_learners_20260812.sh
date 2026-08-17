#!/usr/bin/env bash
set -Eeuo pipefail

profile=${1:?d14 or d15 required}
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
case "$profile" in
  d14)
    worktree=/homes/lzhang/worktrees/experiment7-async-4c45f89
    chains=(a05_raging_bolt_ogerpon_kangaskhan mega_lucario_ex a08_dipplin_seaking)
    ;;
  d15)
    worktree=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
    chains=(a02_submission4_grimmsnarl_froslass_munkidori)
    ;;
  *) exit 2 ;;
esac

declare -A pids
for chain in "${chains[@]}"; do
  matches=$(pgrep -f "^/homes/lzhang/mypath/new/envs/trans/bin/python .*${worktree}/experiment7/integration/run_async_ppo_learner.py .*--chain ${chain}( |$)" || true)
  [[ $(wc -w <<<"$matches") -le 1 ]] || { echo "REFUSE_MULTIPLE chain=$chain pids=$matches"; exit 3; }
  pids[$chain]="$matches"
done

# Let any in-flight GPU update finish.  Polling learners have no child.
for _ in $(seq 1 180); do
  busy=0
  for chain in "${chains[@]}"; do
    pid=${pids[$chain]}
    [[ -n "$pid" ]] || continue
    pgrep -P "$pid" >/dev/null 2>&1 && busy=1
  done
  [[ $busy -eq 0 ]] && break
  sleep 5
done
for chain in "${chains[@]}"; do
  pid=${pids[$chain]}
  [[ -n "$pid" ]] || continue
  pgrep -P "$pid" >/dev/null 2>&1 && { echo "DEFER_LEARNER_BUSY chain=$chain pid=$pid"; exit 4; }
done

for chain in "${chains[@]}"; do
  pid=${pids[$chain]}
  [[ -n "$pid" ]] || continue
  kill "$pid"
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  kill -0 "$pid" 2>/dev/null && { echo "OLD_LEARNER_STILL_RUNNING chain=$chain pid=$pid"; exit 5; }
done

/bin/bash /homes/lzhang/start_adjusted_async_ppo_learners.sh "$profile"
echo "ADAPTIVE_LEARNERS_READY profile=$profile host=$(hostname)"
