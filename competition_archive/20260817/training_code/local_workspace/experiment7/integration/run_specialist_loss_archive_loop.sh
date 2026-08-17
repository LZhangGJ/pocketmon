#!/usr/bin/env bash
set -euo pipefail

MAIN="${EXPERIMENT7_MAIN:-/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811}"
SOURCE_ROOT="$MAIN/buffer/ready"
ARCHIVE_ROOT="$MAIN/monitoring/specialist-loss-replays"
ARCHIVER="${SPECIALIST_LOSS_ARCHIVER:-$MAIN/control/archive_specialist_ppo_losses.py}"
INTERVAL_SECONDS="${SPECIALIST_LOSS_ARCHIVE_INTERVAL_SECONDS:-1800}"
CHAINS=(
  a02_grim_g247
  a02_grim_g247_pokegear
  a08_rabsca
  a08_maxbelt
  lucario_gold_exact
)

mkdir -p "$ARCHIVE_ROOT"
printf '%s\n' "$$" > "$ARCHIVE_ROOT/controller.pid"
trap 'rm -f "$ARCHIVE_ROOT/controller.pid"' EXIT

while true; do
  exec 9>"$ARCHIVE_ROOT/controller.lock"
  if flock -n 9; then
    pids=()
    for chain in "${CHAINS[@]}"; do
      nice -n 10 ionice -c2 -n7 python3 "$ARCHIVER" \
        --source-root "$SOURCE_ROOT" \
        --archive-root "$ARCHIVE_ROOT" \
        --chains "$chain" \
        --keep-generations 3 \
        --max-episodes-per-chain 100 \
        > "$ARCHIVE_ROOT/$chain.loop.log" 2>&1 &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
      wait "$pid" || failed=1
    done
    if [[ "$failed" -eq 0 ]]; then
      python3 "$ARCHIVER" \
        --source-root "$SOURCE_ROOT" \
        --archive-root "$ARCHIVE_ROOT" \
        --chains "${CHAINS[@]}" \
        --summary-only \
        > "$ARCHIVE_ROOT/controller-latest.log" 2>&1
    fi
    printf '{"updatedAt":"%s","lastPassFailed":%s,"nextPassAfterSeconds":%s}\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$failed" "$INTERVAL_SECONDS" \
      > "$ARCHIVE_ROOT/controller-state.json.tmp"
    mv "$ARCHIVE_ROOT/controller-state.json.tmp" "$ARCHIVE_ROOT/controller-state.json"
  fi
  exec 9>&-
  sleep "$INTERVAL_SECONDS"
done
