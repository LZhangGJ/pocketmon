#!/usr/bin/env bash
set -Eeuo pipefail

source_manifest=/suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/windows/2026-08-12/tensordict-sources.json
stage=/dev/shm/lzhang-bc-0812-tensordict-20260813
stage_cache="$stage/tensordict-cache"
stage_manifest="$stage/tensordict-sources-local.json"
reference=/tmp/experiment7-bc-reference-20260813
runtime=/tmp/experiment7-async-bc-runtime-20260813
controller="$runtime/run_async_bc_profile_controller_20260813.py"
python=/homes/lzhang/mypath/new/envs/trans/bin/python
output=/tmp/lzhang-bc-0812-async-20260813
log_root="$output/control"
mkdir -p "$stage_cache" "$log_root"

exec 9>"$stage/stage.lock"
flock -n 9 || { echo STAGE_ALREADY_RUNNING; exit 0; }

mem_available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
shm_available_kib=$(df -Pk /dev/shm | awk 'NR==2 {print $4}')
minimum_mem_kib=$((220 * 1024 * 1024))
minimum_shm_kib=$((100 * 1024 * 1024))
if (( mem_available_kib < minimum_mem_kib || shm_available_kib < minimum_shm_kib )); then
  echo "RAM_GUARD_BLOCK memAvailableKiB=$mem_available_kib shmAvailableKiB=$shm_available_kib"
  exit 2
fi
echo "RAM_GUARD_PASS memAvailableKiB=$mem_available_kib shmAvailableKiB=$shm_available_kib"

for day in 2026-08-{03,04,05,06,07,08,09,10,11,12}; do
  source_dir="/suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/tensordict-cache/$day"
  target_dir="$stage_cache/$day"
  test -f "$source_dir/features_tensordict/meta.json"
  mkdir -p "$target_dir"
  ionice -c2 -n7 nice -n10 rsync -a --delete "$source_dir/" "$target_dir/"
  test -f "$target_dir/features_tensordict/meta.json"
  echo "RAM_DAY_READY day=$day bytes=$(du -sb "$target_dir" | awk '{print $1}')"
done

/usr/bin/python3 - "$source_manifest" "$stage_manifest" "$stage_cache" "$reference" <<'PY'
import json, os, pathlib, sys
source, output, cache, reference = sys.argv[1:]
payload = json.load(open(source, encoding="utf-8"))
payload["referenceRoot"] = str(pathlib.Path(reference).resolve())
for row in payload["datasets"]:
    root = pathlib.Path(cache) / row["name"]
    row["features"] = str(root / "features_tensordict")
    row["tokenCache"] = str(root / "token_cache")
    row["sequenceCache"] = str(root / "sequence_cache")
    row["identityCache"] = str(root / "identity_cache")
    for key in ("features", "tokenCache", "sequenceCache", "identityCache"):
        if not pathlib.Path(row[key]).exists():
            raise SystemExit(f"RAM_STAGE_MISSING {key}={row[key]}")
temporary = output + f".{os.getpid()}.tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    handle.write("\n")
os.replace(temporary, output)
print(json.dumps({"status": "RAM_MANIFEST_READY", "output": output, "datasets": len(payload["datasets"])}))
PY

launch_profile() {
  local profile=$1 train_gpu=$2 validation_gpu=$3 start_epoch=$4 previous=$5
  nohup "$python" -s "$controller" \
    --profile "$profile" --train-gpu "$train_gpu" --validation-gpu "$validation_gpu" \
    --start-epoch "$start_epoch" --previous-root "$previous" --output-root "$output" \
    --sources "$stage_manifest" >"$log_root/$profile-controller.log" 2>&1 </dev/null &
  echo "BC_0812_STARTED profile=$profile pid=$! trainGpu=$train_gpu validationGpu=$validation_gpu"
}

launch_profile standard_1m 3 1 1 /tmp/lzhang-bc-capacity-a100-20260813/standard_1m-persistent-continuation
launch_profile large_256x6 7 1 1 /tmp/lzhang-bc-capacity-a100-20260813/large_256x6-persistent-continuation
date -Iseconds >"$stage/SUCCESS"
echo BC_0812_BOTH_LAUNCHED
