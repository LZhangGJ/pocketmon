#!/usr/bin/env bash
set -Eeuo pipefail

branch_root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-deck-branches-20260812
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-deck-branch-evals-20260813
live_root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
worktree=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
python=/homes/lzhang/mypath/new/envs/trans/bin/python
exporter=/homes/lzhang/worktrees/experiment7-async-4c45f89/experiment7/integration/export_and_package.py
prepare=/homes/lzhang/prepare_a08_deck_branch_evals_20260813.py
runner=/homes/lzhang/run_latest_ppo_full_eval.py
run_shard=/homes/lzhang/run_isolated_arena_shard.sh
bc=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/final_7d/downstream-e3cb2936afb5/seed-20260812/universal_bc.npz
template=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/learners/a08_dipplin_seaking/generation-000333/deployment/packages/live_a08_dipplin_seaking_g000333__a08_dipplin_seaking
hosts=10.113.13.53,10.113.13.54,10.113.13.57,10.113.13.63,10.113.13.64,10.113.13.67,10.113.13.68,10.113.13.69,10.113.13.71,10.113.13.72,10.113.13.73,10.113.13.74,10.113.13.75,10.113.13.77,10.113.13.78

prepare_one() {
  local name=$1 generation=$2 candidate
  candidate=$(printf '%s/%s_g%04d' "$root" "$name" "$generation")
  if [[ -s "$candidate/state/league.json" ]]; then return; fi
  "$python" -s "$prepare" \
    --branch-root "$branch_root" --output-root "$root" \
    --live-league "$live_root/state/league.json" --template-agent "$template" \
    --python "$python" --exporter "$exporter" --candidate "$name" "$generation"
}

run_one() {
  local name=$1 generation=$2 candidate latest
  candidate=$(printf '%s/%s_g%04d' "$root" "$name" "$generation")
  latest="$candidate/monitoring/full-matrix/latest.json"
  if [[ -s "$latest" ]] && grep -q '"status": "complete"' "$latest"; then
    echo "A08_BRANCH_EVAL_SKIP_COMPLETE branch=$name generation=$generation"
    return
  fi
  prepare_one "$name" "$generation"
  echo "A08_BRANCH_EVAL_START branch=$name generation=$generation at=$(date -Iseconds)"
  PYTHONNOUSERSITE=1 "$python" -s "$runner" \
    --league-root "$candidate" --worktree "$worktree" --python "$python" \
    --run-shard "$run_shard" --bc-portable "$bc" \
    --games-per-frozen 40 --games-per-head-to-head 40 --shards 45 \
    --distributed-hosts "$hosts" --max-shards-per-host 3 \
    >"$root/logs/${name}_g$(printf '%04d' "$generation").log" 2>&1
  echo "A08_BRANCH_EVAL_DONE branch=$name generation=$generation at=$(date -Iseconds)"
}

run_all() {
  mkdir -p "$root/logs"
  run_one a08_maxbelt 20
  run_one a08_lilligant 20
  while [[ ! -s "$branch_root/a08_lilligant_maxbelt/generation-0015/checkpoint.pt" || ! -s "$branch_root/a08_lilligant_maxbelt/generation-0015/metrics.json" ]]; do
    echo "WAIT_A08_COMBO_G15 at=$(date -Iseconds)"
    sleep 30
  done
  run_one a08_lilligant_maxbelt 15
  touch "$root/POOL40_COMPLETE"
}

if [[ "${1:-}" == --run ]]; then run_all; exit; fi
mkdir -p "$root"
pidfile="$root/controller.pid"
if [[ -s "$pidfile" ]]; then
  old=$(<"$pidfile")
  if [[ "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "A08_BRANCH_EVAL_ALREADY_RUNNING pid=$old"
    exit
  fi
fi
nohup /bin/bash "$0" --run >"$root/controller.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
sleep 2
kill -0 "$pid"
echo "A08_BRANCH_EVAL_CONTROLLER_STARTED pid=$pid root=$root"
