#!/usr/bin/env bash
set -u
echo GPU_FREE
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true
echo FILESYSTEMS
df -hT /tmp /dataT0/Free/lzhang 2>/dev/null || true
for path in /local /scratch /data /dataT0/Free/lzhang; do
  if [ -d "$path" ]; then
    echo "PATH=$path"
    df -hT "$path" | tail -1
  fi
done
echo MEMORY
free -h | head -2
echo WORKTREES
for path in /homes/lzhang/worktrees/experiment7-async-4c45f89 /homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0; do
  [ -d "$path" ] && echo PRESENT=$path || echo MISSING=$path
done
echo SOURCE_BYTES
python3 - <<'PY'
import json, os
p='/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/capacity-comparison/control/universal-10d-sources.json'
d=json.load(open(p))
paths=[]
for row in d.get('datasets',[]):
    for key in ('features','tokenCache','sequenceCache','identityCache'):
        q=row.get(key)
        if q and q not in paths: paths.append(q)
total=0
for q in paths:
    try: total += os.path.getsize(q)
    except OSError: pass
print(total, len(paths))
PY
