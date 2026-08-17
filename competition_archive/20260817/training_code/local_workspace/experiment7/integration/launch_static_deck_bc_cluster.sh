#!/usr/bin/env bash
set -euo pipefail

strict_root=${1:?strict root required}
control_root=${2:?control root required}
runtime=/dataT0/Free/lzhang/pocketmon-runs/experiment7-static-deck-bc-10d-20260815/control/runtime
config=$runtime/config/static_deck_bc_10d_20260815.json
python=/homes/lzhang/mypath/new/envs/trans/bin/python
manifest=$strict_root/tensordict-sources.json
mkdir -p "$control_root/launch"

# Largest profile is reserved for the already-smoked Windows RTX 5070 Ti.
# Linux profiles use one GPU each; the distinct validator process shares that
# GPU behind maxPendingValidations=1, so validation and the next epoch do not
# compute concurrently.
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

check_idle() {
  local host=$1 gpu=$2
  ssh "$host" "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i $gpu" | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1>64 || $2>5) exit 1}'
}

while read -r profile host gpu batch; do
  check_idle "$host" "$gpu"
done <<<"$plan"
sleep 10
while read -r profile host gpu batch; do
  check_idle "$host" "$gpu"
  if ssh "$host" "pgrep -af '[r]un_static_deck_bc_profile.py.*--archetype $profile'" >/dev/null; then
    echo "duplicate profile process: $profile on $host" >&2
    exit 3
  fi
done <<<"$plan"

: >"$control_root/launch/linux-profile-pids.tsv"
while read -r profile host gpu batch; do
  local_root=/dev/shm/lzhang-static-deck-bc-10d-20260815/$profile
  pid=$(ssh "$host" "mkdir -p '$local_root/logs'; nohup '$python' -s '$runtime/integration/run_static_deck_bc_profile.py' --config '$config' --strict-manifest '$manifest' --archetype '$profile' --local-root '$local_root' --control-root '$control_root' --runtime-root '$runtime' --device '$gpu' --batch-size '$batch' --python '$python' > '$local_root/logs/controller.log' 2>&1 & echo \$!")
  printf '%s\t%s\t%s\t%s\t%s\n' "$profile" "$host" "$gpu" "$batch" "$pid" >>"$control_root/launch/linux-profile-pids.tsv"
done <<<"$plan"

cat >"$control_root/launch/windows-profile-pending.json" <<EOF
{"profile":"grimmsnarl_froslass_munkidori","host":"windows-rtx5070ti","gpu":0,"status":"reserved_for_windows_post_strict_controller","formalEpochStarted":false}
EOF
cat "$control_root/launch/linux-profile-pids.tsv"
