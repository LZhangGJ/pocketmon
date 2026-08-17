#!/usr/bin/env bash
set -Eeuo pipefail

root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-deck-branch-evals-20260813
worktree=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
python=/homes/lzhang/mypath/new/envs/trans/bin/python
runner=/homes/lzhang/run_latest_ppo_full_eval.py
run_shard=/homes/lzhang/run_isolated_arena_shard.sh
bc=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/final_7d/downstream-e3cb2936afb5/seed-20260812/universal_bc.npz
hosts=10.113.13.53,10.113.13.54,10.113.13.57,10.113.13.63,10.113.13.64,10.113.13.67,10.113.13.68,10.113.13.69,10.113.13.71,10.113.13.72,10.113.13.73,10.113.13.74,10.113.13.75,10.113.13.77,10.113.13.78

run_all() {
  mkdir -p "$root/arch-extra/logs"
  "$python" -s /homes/lzhang/prepare_a08_branch_arch_extra_20260813.py \
    --branch-eval-root "$root" \
    --live-pool /dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/state/opponent-pool-live.json
  for name in a08_maxbelt_g0020 a08_lilligant_g0020 a08_lilligant_maxbelt_g0015; do
    candidate="$root/arch-extra/$name"
    latest="$candidate/monitoring/full-matrix/latest.json"
    if [[ -s "$latest" ]] && grep -q '"status": "complete"' "$latest"; then continue; fi
    echo "ARCH_EXTRA_START candidate=$name at=$(date -Iseconds)"
    PYTHONNOUSERSITE=1 "$python" -s "$runner" \
      --league-root "$candidate" --worktree "$worktree" --python "$python" \
      --run-shard "$run_shard" --bc-portable "$bc" \
      --games-per-frozen 40 --games-per-head-to-head 40 --shards 16 \
      --distributed-hosts "$hosts" --max-shards-per-host 2 \
      >"$root/arch-extra/logs/$name.log" 2>&1
    echo "ARCH_EXTRA_DONE candidate=$name at=$(date -Iseconds)"
  done
  touch "$root/arch-extra/COMPLETE"
}

if [[ "${1:-}" == --run ]]; then run_all; exit; fi
mkdir -p "$root/arch-extra"
pidfile="$root/arch-extra/controller.pid"
if [[ -s "$pidfile" ]]; then
  old=$(<"$pidfile")
  if [[ "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "ARCH_EXTRA_ALREADY_RUNNING pid=$old"
    exit
  fi
fi
nohup /bin/bash "$0" --run >"$root/arch-extra/controller.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
sleep 2
kill -0 "$pid"
echo "ARCH_EXTRA_CONTROLLER_STARTED pid=$pid"
