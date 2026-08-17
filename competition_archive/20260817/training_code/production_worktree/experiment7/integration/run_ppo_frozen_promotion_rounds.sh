#!/usr/bin/env bash
set -euo pipefail

main_root=${1:?main root is required}
python_bin=${PYTHON_BIN:-/homes/lzhang/mypath/new/envs/trans/bin/python}
worktree=${EXPERIMENT7_WORKTREE:-/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0}
promotion_python="$main_root/control/run_ppo_frozen_promotion_eval.py"
promotion_state="$main_root/control/ppo-frozen-promotion-state.json"
tactical_builder="$main_root/control/build_ppo_tactical_guardrail_evidence.py"
tactical_evidence="$main_root/monitoring/ppo-tactical-guardrails/latest.json"
bc_portable=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/0812-d14-ram-npz-fast-20260813/replacement-screening/large_256x6/universal_bc.npz
log="$main_root/monitoring/ppo-frozen-promotion/driver.log"
pidfile="$main_root/control/ppo-frozen-promotion-eval.pid"

mkdir -p "$main_root/monitoring/ppo-frozen-promotion"
if [[ -s "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
  printf 'ALREADY_RUNNING=%s\n' "$(cat "$pidfile")"
  exit 0
fi

run_driver() {
  for round_index in 1 2; do
    PYTHONNOUSERSITE=1 "$python_bin" -s "$promotion_python" \
      --league-root "$main_root" \
      --promotion-state "$promotion_state" \
      --worktree "$worktree" \
      --python "$python_bin" \
      --run-shard /homes/lzhang/run_isolated_arena_shard.sh \
      --bc-portable "$bc_portable" \
      --games-per-frozen 4 \
      --games-per-head-to-head 40 \
      --shards 60 \
      --distributed-hosts 10.113.13.74 \
      --max-shards-per-host 60
    "$python_bin" "$tactical_builder" \
      --buffer-root "$main_root/buffer" \
      --output "$tactical_evidence" \
      --minimum-revision 9
    "$python_bin" "$main_root/control/promote_ppo_to_frozen_pool.py" \
      --league "$main_root/state/league.json" \
      --state "$promotion_state" \
      --source-base-pool "$main_root/state/opponent-pool-base-plus-bc-20260812-dual.json" \
      --output-base-pool "$main_root/state/opponent-pool-base-plus-promoted-ppo.json" \
      --reports-root "$main_root/monitoring/full-matrix" \
      --reports-root "$main_root/monitoring/ppo-frozen-promotion" \
      --min-frozen-games 40 \
      --min-direct-games 40 \
      --max-regression-pp 2 \
      --required-passes 2 \
      --tactical-evidence "$tactical_evidence"
  done
}

export main_root python_bin worktree promotion_python promotion_state bc_portable
export tactical_builder tactical_evidence
nohup bash -c "$(declare -f run_driver); run_driver" >"$log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
printf 'PROMOTION_EVAL_PID=%s\n' "$pid"
