#!/usr/bin/env bash
set -euo pipefail

# The first copy is already being staged to doraemon20's local /tmp disk.  This
# launcher waits for that exact rsync, promotes the immutable cache into tmpfs,
# and starts two bounded-prefetch A100 candidates without touching RTX baselines.
disk_stage=/tmp/lzhang-bc-capacity-a100-20260812
ram_stage=/dev/shm/lzhang-bc-capacity-a100-20260812
source_manifest=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/capacity-comparison/control/universal-10d-sources.json
ram_manifest="$ram_stage/universal-10d-sources-ram.json"
output_root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/capacity-comparison-a100-ram-prefetch-b256
python=/homes/lzhang/mypath/new/envs/trans/bin/python
trainer=/homes/lzhang/worktrees/experiment7-async-4c45f89/experiment7/integration/train_universal_bc.py
copy_pid=${1:-1365404}

available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
shm_available_kib=$(df -Pk /dev/shm | awk 'NR==2 {print $4}')
# Reserve at least 240 GiB before staging: 38 GiB tmpfs, two approximately
# 82-GiB feature bundles, bounded pinned queues, and a generous system margin.
minimum_available_kib=$((240 * 1024 * 1024))
minimum_shm_kib=$((48 * 1024 * 1024))
if (( available_kib < minimum_available_kib )); then
  echo "RAM_GUARD_BLOCK availableKiB=$available_kib requiredKiB=$minimum_available_kib"
  exit 2
fi
if (( shm_available_kib < minimum_shm_kib )); then
  echo "SHM_GUARD_BLOCK availableKiB=$shm_available_kib requiredKiB=$minimum_shm_kib"
  exit 2
fi
echo "RAM_GUARD_PASS availableKiB=$available_kib shmAvailableKiB=$shm_available_kib"

while kill -0 "$copy_pid" 2>/dev/null; do
  echo "WAIT_DISK_STAGE pid=$copy_pid bytes=$(du -sb "$disk_stage/root" 2>/dev/null | awk '{print $1}')"
  sleep 20
done

mkdir -p "$ram_stage/root" "$output_root/logs"
ionice -c2 -n7 nice -n 10 rsync -a "$disk_stage/root/" "$ram_stage/root/"
python3 - "$source_manifest" "$ram_manifest" "$ram_stage/root" <<'PY'
import json, os, sys
source, output, root = sys.argv[1:]
payload = json.load(open(source))
missing = []
for row in payload.get("datasets", []):
    for key in ("features", "tokenCache", "sequenceCache", "identityCache"):
        value = row.get(key)
        if value and value.startswith("/"):
            row[key] = root.rstrip("/") + value
        if row.get(key) and not os.path.exists(row[key]):
            missing.append(row[key])
if missing:
    raise SystemExit("RAM_STAGE_MISSING " + repr(missing[:5]))
with open(output, "w") as handle:
    json.dump(payload, handle, separators=(",", ":"))
print("RAM_STAGE_VALID")
PY

launch_one() {
  local name=$1 gpu=$2 dmodel=$3 heads=$4 layers=$5 ffdim=$6
  local out="$output_root/$name"
  while :; do
    local used
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
    if (( used <= 1024 )); then break; fi
    echo "WAIT_GPU index=$gpu usedMiB=$used"
    sleep 20
  done
  mkdir -p "$out"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 \
    nohup ionice -c2 -n7 nice -n 10 "$python" -s "$trainer" \
      --sources "$ram_manifest" --output-dir "$out" --device cuda:0 \
      --seed 20260812 --epochs 4 --batch-size 256 --learning-rate 2e-4 \
      --weight-decay 1e-4 --value-loss-weight 0.05 \
      --prefetch-batches 6 --prefetch-workers 2 \
      --d-model "$dmodel" --heads "$heads" --layers "$layers" \
      --ff-dim "$ffdim" --dropout 0.05 \
      >"$output_root/logs/$name.log" 2>&1 </dev/null &
  echo "$!" >"$out.pid"
  echo "RAM_PREFETCH_STARTED name=$name gpu=$gpu pid=$!"
}

launch_one large_256x6 3 256 8 6 1024
launch_one standard_1m 1 128 4 3 384
echo RAM_PREFETCH_LAUNCHED
