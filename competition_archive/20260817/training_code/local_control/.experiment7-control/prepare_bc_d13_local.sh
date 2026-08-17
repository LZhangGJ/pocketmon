#!/usr/bin/env bash
set -euo pipefail

LOCAL=/tmp/experiment7-bc-d13-local-20260813
SOURCE_MANIFEST=/suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/windows/2026-08-12/tensordict-sources.json
SHARED_BASE=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/capacity-comparison-a100-ram-prefetch-b256
mkdir -p "${LOCAL}/reference" "${LOCAL}/standard_1m" "${LOCAL}/large_256x6"
rsync -a /homes/lzhang/worktrees/experiment7-async-4c45f89/experiment7/reference/ "${LOCAL}/reference/"
for profile in standard_1m large_256x6; do
  rsync -a "${SHARED_BASE}/${profile}/best_model.pt" "${LOCAL}/${profile}/best_model.pt"
  rsync -a "${SHARED_BASE}/${profile}/training_report.json" "${LOCAL}/${profile}/training_report.json"
done
export LOCAL SOURCE_MANIFEST
/homes/lzhang/mypath/new/envs/trans/bin/python -s - <<'PY'
import json
import os
from pathlib import Path
p = json.loads(Path(os.environ["SOURCE_MANIFEST"]).read_text(encoding="utf-8"))
p["referenceRoot"] = str(Path(os.environ["LOCAL"]) / "reference")
dst = Path(os.environ["LOCAL"]) / "tensordict-sources.json"
dst.write_text(json.dumps(p, separators=(",", ":")) + "\n", encoding="utf-8")
PY
echo D13_LOCAL_PREPARED
