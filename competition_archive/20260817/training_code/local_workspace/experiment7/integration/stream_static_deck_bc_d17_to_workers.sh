#!/usr/bin/env bash
set -euo pipefail

build_root=${1:-/dev/shm/lzhang-static-deck-bc-10d-20260815-build}
runtime=${2:-/dataT0/Free/lzhang/pocketmon-runs/experiment7-static-deck-bc-10d-20260815/control/runtime}
control=${3:-/dataT0/Free/lzhang/pocketmon-runs/experiment7-static-deck-bc-10d-20260815/control}
python=${PYTHON_BIN:-/homes/lzhang/mypath/new/envs/trans/bin/python}
authority_manifest=/tmp/lzhang-strict-scoregt1000-window-20260815T1200Z/tensordict-sources.json
initializer=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/learners/universal_ppo_large_256x6/generation-000051/checkpoint.pt
stream_control=$build_root/stream-control
mkdir -p "$stream_control"

plan=$(cat <<'EOF'
dragapult_munkidori doraemon03 0 512
alakazam_dudunsparce doraemon03 1 512
mega_lopunny_mega_froslass doraemon15 0 512
hydrapple_dipplin_ogerpon doraemon15 1 512
crustle_mega_kangaskhan doraemon15 2 512
ogerpon_only doraemon15 3 512
mega_lucario_hariyama doraemon12 0 512
festival_dipplin doraemon12 1 512
slowking_mega_kangaskhan doraemon12 2 512
raging_bolt_ogerpon_kangaskhan doraemon04 0 256
EOF
)

gpu_idle() {
  local host=$1 gpu=$2
  ssh "$host" "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i '$gpu'" |
    awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1>64 || $2>5) exit 1}'
}

launch_when_ready() {
  local profile=$1 host=$2 gpu=$3 batch=$4
  local source=$build_root/profiles/$profile
  local target=/dev/shm/lzhang-static-deck-bc-10d-20260815/$profile
  local record=$stream_control/$profile
  mkdir -p "$record"
  while [[ ! -f "$source/tensordict-sources.json" ]]; do
    if [[ -f "$build_root/pids/$profile.pid" ]] &&
       ! kill -0 "$(cat "$build_root/pids/$profile.pid")" 2>/dev/null; then
      printf 'builder_exited_before_completion\n' >"$record/failed"
      return 1
    fi
    sleep 10
  done
  printf 'transferring\n' >"$record/status"
  ssh "$host" "mkdir -p '$(dirname "$target")'"
  rsync -a --sparse --info=progress2 "$source/" "$host:$target/" >"$record/rsync.log" 2>&1
  rsync -a "$authority_manifest" "$host:$target/strict-authority-manifest.json"
  rsync -a "$initializer" "$host:$target/initializer.pt"
  printf 'verifying_gpu\n' >"$record/status"
  gpu_idle "$host" "$gpu"
  sleep 10
  gpu_idle "$host" "$gpu"
  if ssh "$host" "pgrep -af '[r]un_static_deck_bc_profile.py.*--archetype $profile'" >/dev/null; then
    printf 'duplicate_profile\n' >"$record/failed"
    return 1
  fi
  pid=$(ssh "$host" "mkdir -p '$target/logs'; nohup '$python' -s '$runtime/integration/run_static_deck_bc_profile.py' --config '$runtime/config/static_deck_bc_10d_20260815.json' --strict-manifest '$target/strict-authority-manifest.json' --archetype '$profile' --local-root '$target' --control-root '$control' --runtime-root '$runtime' --device '$gpu' --batch-size '$batch' --python '$python' >'$target/logs/controller.log' 2>&1 & echo \$!")
  printf '%s\n' "$pid" >"$record/profile-controller.pid"
  printf 'launched\n' >"$record/status"
}

while read -r profile host gpu batch; do
  launch_when_ready "$profile" "$host" "$gpu" "$batch" &
  printf '%s\n' "$!" >"$stream_control/$profile.watcher.pid"
done <<<"$plan"
wait
