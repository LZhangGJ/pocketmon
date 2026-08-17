#!/usr/bin/env bash
set -euo pipefail

main_root=${1:?main root is required}
python_bin=${PYTHON_BIN:-python3}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
interval_seconds=${PROMOTION_POLL_SECONDS:-1800}
tactical_evidence="$main_root/monitoring/ppo-tactical-guardrails/latest.json"

while true; do
  "$python_bin" "$script_dir/build_ppo_tactical_guardrail_evidence.py" \
    --buffer-root "$main_root/buffer" \
    --output "$tactical_evidence" \
    --minimum-revision 9
  "$python_bin" "$script_dir/promote_ppo_to_frozen_pool.py" \
    --league "$main_root/state/league.json" \
    --state "$main_root/control/ppo-frozen-promotion-state.json" \
    --source-base-pool "$main_root/state/opponent-pool-base-plus-bc-20260812-dual.json" \
    --output-base-pool "$main_root/state/opponent-pool-base-plus-promoted-ppo.json" \
    --reports-root "$main_root/monitoring/full-matrix" \
    --reports-root "$main_root/monitoring/ppo-frozen-promotion" \
    --min-frozen-games 40 \
    --min-direct-games 40 \
    --max-regression-pp 2 \
    --required-passes 2 \
    --tactical-evidence "$tactical_evidence"
  sleep "$interval_seconds"
done
