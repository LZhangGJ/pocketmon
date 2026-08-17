#!/usr/bin/env bash
set -euo pipefail

# Build the immutable specialist shards next to the authoritative strict
# window.  This is intentionally build-only: each completed profile is copied
# to its assigned worker before that worker starts its own trainer/validator.
strict_root=${1:-/tmp/lzhang-strict-scoregt1000-window-20260815T1200Z}
build_root=${2:-/dev/shm/lzhang-static-deck-bc-10d-20260815-build}
runtime=${3:-/dataT0/Free/lzhang/pocketmon-runs/experiment7-static-deck-bc-10d-20260815/control/runtime}
expected_manifest_sha=${EXPECTED_MANIFEST_SHA:-b80edd05263002dbaf830dc4ad0296b7ede435f7a861461a9f28838a72d5010b}
python=${PYTHON_BIN:-/homes/lzhang/mypath/new/envs/trans/bin/python}
manifest=$strict_root/tensordict-sources.json
config=$runtime/config/static_deck_bc_10d_20260815.json
control=$build_root/control

test -f "$strict_root/SUCCESS"
test -f "$manifest"
actual_manifest_sha=$(sha256sum "$manifest" | awk '{print $1}')
test "$actual_manifest_sha" = "$expected_manifest_sha"
mkdir -p "$build_root/pids" "$build_root/logs" "$control"

profiles=(
  dragapult_munkidori
  grimmsnarl_froslass_munkidori
  alakazam_dudunsparce
  mega_lopunny_mega_froslass
  hydrapple_dipplin_ogerpon
  crustle_mega_kangaskhan
  slowking_mega_kangaskhan
  mega_lucario_hariyama
  ogerpon_only
  festival_dipplin
  raging_bolt_ogerpon_kangaskhan
)

for profile in "${profiles[@]}"; do
  profile_root=$build_root/profiles/$profile
  pid_file=$build_root/pids/$profile.pid
  if [[ -f "$profile_root/tensordict-sources.json" ]] &&
     [[ -f "$control/profiles/$profile/completion.json" ]]; then
    continue
  fi
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    continue
  fi
  mkdir -p "$profile_root/logs"
  nohup "$python" -s "$runtime/integration/run_static_deck_bc_profile.py" \
    --config "$config" \
    --strict-manifest "$manifest" \
    --archetype "$profile" \
    --local-root "$profile_root" \
    --control-root "$control" \
    --runtime-root "$runtime" \
    --device 0 \
    --batch-size 512 \
    --python "$python" \
    --build-only \
    >"$build_root/logs/$profile.log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$pid_file"
  printf '%s\t%s\t%s\n' "$profile" "$pid" "$profile_root"
done
