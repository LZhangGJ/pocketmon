#!/usr/bin/env bash
set -Eeuo pipefail

host=$(hostname -s)
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
league=$root/state/league.json
run_root=$root/learners
buffer_root=$root/buffer
python=/homes/lzhang/mypath/new/envs/trans/bin/python
clean=/homes/lzhang/worktrees/experiment7-a08-learner-clean-8885a59
logs=$root/logs/clean-worktree-recovery-20260813
receipts=$root/monitoring/clean-worktree-recovery

mkdir -p "$logs" "$receipts"
test -x "$python"
test -f "$clean/experiment7/integration/train_universal_ppo.py"
if [[ -n "$(git -C "$clean" status --porcelain)" ]]; then
  echo "CLEAN_WORKTREE_DIRTY path=$clean" >&2
  exit 2
fi

start_one() {
  local chain=$1 launcher=$2
  shift 2
  local matches
  matches=$(pgrep -f "run_async_ppo_learner.py .*--chain $chain( |$)" || true)
  if [[ -n "$matches" ]]; then
    for pid in $matches; do
      if ps -p "$pid" -o args= | grep -q -- "--worktree $clean" \
        && ps -p "$pid" -o args= | grep -q -- "--deployment-staging-root /dev/shm/experiment7-ppo-deploy-staging"; then
        echo "ALREADY_RECOVERED chain=$chain pid=$pid"
        return
      fi
    done
    for pid in $matches; do
      echo "STOP_OLD chain=$chain pid=$pid"
      children=$(pgrep -P "$pid" || true)
      if [[ -n "$children" ]]; then
        echo "STOP_CHILDREN chain=$chain parent=$pid children=$children"
        kill -TERM $children
      fi
      kill -TERM "$pid"
    done
    for _ in $(seq 1 20); do
      local alive=0
      for pid in $matches; do kill -0 "$pid" 2>/dev/null && alive=1; done
      [[ $alive -eq 0 ]] && break
      sleep 1
    done
    for pid in $matches; do
      if kill -0 "$pid" 2>/dev/null; then
        echo "OLD_DID_NOT_STOP chain=$chain pid=$pid" >&2
        exit 3
      fi
    done
  fi
  local log="$logs/${chain}-${host}.log"
  nohup "$python" "$launcher/experiment7/integration/run_async_ppo_learner.py" \
    --league "$league" --chain "$chain" --worktree "$clean" \
    --run-root "$run_root" --buffer-root "$buffer_root" --python "$python" \
    --deployment-staging-root /dev/shm/experiment7-ppo-deploy-staging \
    --device cuda:0 --max-behavior-lag 2 "$@" --poll-seconds 10 \
    >"$log" 2>&1 </dev/null &
  local pid=$!
  sleep 2
  kill -0 "$pid"
  printf '%s\n' "$pid" >"$receipts/${chain}-${host}.pid"
  printf '{"chain":"%s","host":"%s","pid":%s,"launcher":"%s","formalWorktree":"%s","startedAt":"%s"}\n' \
    "$chain" "$host" "$pid" "$launcher" "$clean" "$(date -Iseconds)" \
    >"$receipts/${chain}-${host}.json"
  echo "RECOVERED chain=$chain pid=$pid log=$log"
}

case "$host" in
  doraemon14)
    launcher=/homes/lzhang/worktrees/experiment7-async-4c45f89
    start_one a05_raging_bolt_ogerpon_kangaskhan "$launcher" \
      --min-decisions 5000 --teacher-anchor-coefficient 0.02 --seat1-weight 1.0
    start_one mega_lucario_ex "$launcher" \
      --min-decisions 6000 --teacher-anchor-coefficient 0.04 --seat1-weight 1.0
    start_one a08_dipplin_seaking "$launcher" \
      --min-decisions 4000 --teacher-anchor-coefficient 0.02 --seat1-weight 2.0 \
      --normalize-advantages-by-player --balance-player-minibatches
    ;;
  doraemon15)
    launcher=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
    start_one a02_submission4_grimmsnarl_froslass_munkidori "$launcher" \
      --min-decisions 4000 --teacher-anchor-coefficient 0.02 --seat1-weight 1.0
    ;;
  *)
    echo "UNEXPECTED_HOST host=$host" >&2
    exit 4
    ;;
esac
