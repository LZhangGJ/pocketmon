#!/usr/bin/env bash
set -euo pipefail

wait_pid=${1:-0}
source_root=/dataT0/Free/lzhang/pocketmon-runs/replay-refresh-20260812
target_root=/homes/lzhang/pocketmon/runs/replay-refresh-20260812
source_manifest=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/capacity-comparison/control/universal-10d-sources.json
target_manifest="$target_root/control/universal-10d-sources-homes-tier.json"
log_prefix=HOMES_CACHE_MIGRATION

if (( wait_pid > 0 )); then
  while kill -0 "$wait_pid" 2>/dev/null; do
    echo "$log_prefix WAIT_SOURCE_COPY pid=$wait_pid"
    sleep 30
  done
fi

while :; do
  io=$(awk '/^some / {for(i=1;i<=NF;i++) if($i ~ /^avg10=/){split($i,a,"="); print a[2]}}' /proc/pressure/io)
  cpu=$(awk -v n="$(nproc)" '{printf "%.2f", 100*$1/n}' /proc/loadavg)
  if python3 - "$cpu" "$io" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) < 95 and float(sys.argv[2]) < 80 else 1)
PY
  then
    echo "$log_prefix GUARD_PASS cpu=$cpu io=$io"
    break
  fi
  echo "$log_prefix GUARD_WAIT cpu=$cpu io=$io"
  sleep 30
done

mkdir -p "$target_root/cache" "$target_root/control"
for day in 2026-08-09 2026-08-10 2026-08-11; do
  src="$source_root/cache/$day/prepared/universal"
  dst="$target_root/cache/$day/prepared/universal"
  test -d "$src"
  mkdir -p "$dst"
  echo "$log_prefix COPY_BEGIN day=$day"
  ionice -c2 -n7 nice -n 10 rsync -a --partial "$src/" "$dst/"
  python3 - "$src" "$dst" "$day" <<'PY'
import os, sys

def inventory(root):
    files = total = 0
    for base, _, names in os.walk(root):
        for name in names:
            path = os.path.join(base, name)
            files += 1
            total += os.path.getsize(path)
    return files, total

source, target, day = sys.argv[1:]
left, right = inventory(source), inventory(target)
if left != right:
    raise SystemExit(f"CACHE_VALIDATION_FAILED day={day} source={left} target={right}")
print(f"CACHE_VALIDATION_PASS day={day} files={left[0]} bytes={left[1]}")
PY
done

python3 - "$source_manifest" "$target_manifest" "$source_root" "$target_root" <<'PY'
import json, os, sys

source, output, old_root, new_root = sys.argv[1:]
payload = json.load(open(source))
wanted = {"2026-08-09", "2026-08-10", "2026-08-11"}
for row in payload.get("datasets", []):
    if row.get("name") not in wanted:
        continue
    for key, value in list(row.items()):
        if isinstance(value, str) and value.startswith(old_root + "/"):
            row[key] = new_root + value[len(old_root):]
    for key in ("features", "tokenCache", "sequenceCache", "identityCache"):
        if not os.path.exists(row[key]):
            raise SystemExit(f"MANIFEST_TARGET_MISSING {row['name']} {key} {row[key]}")
with open(output, "w") as handle:
    json.dump(payload, handle, separators=(",", ":"))
print(f"HOMES_TIER_MANIFEST_READY path={output}")
PY

touch "$target_root/control/MIGRATION_COMPLETE"
echo "$log_prefix COMPLETE manifest=$target_manifest"
