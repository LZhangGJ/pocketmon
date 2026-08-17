#!/usr/bin/env bash
set -euo pipefail

shared=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812
default_manifest="$shared/capacity-comparison/control/universal-10d-sources.json"
homes_manifest=/homes/lzhang/pocketmon/runs/replay-refresh-20260812/control/universal-10d-sources-homes-tier.json
source_manifest="$default_manifest"
if [ -f "$homes_manifest" ]; then
  source_manifest="$homes_manifest"
  echo "USING_HOMES_TIER_MANIFEST path=$source_manifest"
fi
stage=/tmp/lzhang-bc-capacity-a100-20260812
stage_root="$stage/root"
staged_manifest="$stage/universal-10d-sources-local.json"
output_root="$shared/capacity-comparison-a100-b256"
python=/homes/lzhang/mypath/new/envs/trans/bin/python
trainer=/homes/lzhang/worktrees/experiment7-async-4c45f89/experiment7/integration/train_universal_bc.py

if pgrep -f '[t]rain_universal_bc.py.*capacity-comparison-a100-b256' >/dev/null; then
  echo A100_FASTTRACK_ALREADY_RUNNING
  exit 0
fi

cpu=$(awk -v n="$(nproc)" '{printf "%.2f", 100*$1/n}' /proc/loadavg)
io=$(awk '/^some / {for(i=1;i<=NF;i++) if($i ~ /^avg10=/){split($i,a,"="); print a[2]}}' /proc/pressure/io)
python3 - "$cpu" "$io" <<'PY'
import sys
cpu=float(sys.argv[1]); io=float(sys.argv[2])
if cpu >= 95 or io >= 80:
    raise SystemExit(f'RESOURCE_GUARD_BLOCK cpu={cpu:.2f} io={io:.2f}')
print(f'RESOURCE_GUARD_PASS cpu={cpu:.2f} io={io:.2f}')
PY

for gpu in 1 3; do
  read -r util used <<<"$(nvidia-smi -i "$gpu" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ',')"
  if [ "$used" -gt 1024 ]; then
    echo "GPU_NOT_FREE index=$gpu util=$util usedMiB=$used"
    exit 2
  fi
done

mkdir -p "$stage_root" "$output_root/logs"
python3 - "$source_manifest" "$stage/paths.txt" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
paths=[]
for row in d.get('datasets',[]):
    for key in ('features','tokenCache','sequenceCache','identityCache'):
        value=row.get(key)
        if value and value not in paths:
            paths.append(value)
with open(sys.argv[2],'w') as f:
    for path in paths:
        f.write(path.lstrip('/')+'\n')
print(f'STAGE_FILES count={len(paths)}')
PY

# --files-from changes rsync's recursion defaults: -a alone only creates the
# listed cache directories.  Keep the absolute-path layout under stage_root
# and explicitly recurse so the cache payload is actually copied.
ionice -c2 -n7 nice -n 10 rsync -aRr --files-from="$stage/paths.txt" / "$stage_root/"
python3 - "$source_manifest" "$staged_manifest" "$stage_root" <<'PY'
import json, sys
source, output, root = sys.argv[1:]
d=json.load(open(source))
for row in d.get('datasets',[]):
    for key in ('features','tokenCache','sequenceCache','identityCache'):
        value=row.get(key)
        if value and value.startswith('/'):
            row[key]=root.rstrip('/')+value
with open(output,'w') as f:
    json.dump(d,f,separators=(',',':'))
PY
python3 - "$staged_manifest" <<'PY'
import json, os, sys
d=json.load(open(sys.argv[1]))
missing=[]
for row in d.get('datasets',[]):
    for key in ('features','tokenCache','sequenceCache','identityCache'):
        value=row.get(key)
        if value and not os.path.exists(value): missing.append(value)
if missing: raise SystemExit('MISSING_STAGED_FILES '+repr(missing[:5]))
print('STAGE_VALID')
PY

launch_one() {
  local name=$1 gpu=$2 dmodel=$3 heads=$4 layers=$5 ffdim=$6
  local out="$output_root/$name"
  mkdir -p "$out"
  env CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 \
    nohup ionice -c2 -n7 nice -n 10 "$python" -s "$trainer" \
      --sources "$staged_manifest" --output-dir "$out" --device cuda:0 \
      --seed 20260812 --epochs 4 --batch-size 256 --learning-rate 2e-4 \
      --weight-decay 1e-4 --value-loss-weight 0.05 \
      --d-model "$dmodel" --heads "$heads" --layers "$layers" --ff-dim "$ffdim" --dropout 0.05 \
      >"$output_root/logs/$name.log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" > "$output_root/$name.pid"
  echo "A100_FASTTRACK_STARTED name=$name gpu=$gpu pid=$pid batch=256"
}

launch_one standard_1m 1 128 4 3 384
launch_one large_256x6 3 256 8 6 1024
sleep 5
for name in standard_1m large_256x6; do
  pid=$(cat "$output_root/$name.pid")
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "A100_FASTTRACK_EARLY_EXIT name=$name"
    tail -40 "$output_root/logs/$name.log" || true
    exit 3
  fi
done
echo A100_FASTTRACK_READY
