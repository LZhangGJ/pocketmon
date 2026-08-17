#!/usr/bin/env bash
set -Eeuo pipefail

league=/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811
worktree=/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0
python=/homes/lzhang/mypath/new/envs/trans/bin/python
cg_dir=/dataT0/Free/lzhang/pocketmon-runs/experiment7-opponent-pool-20260810/official-engine/cg
a02_root="$league/learners/a02_submission4_grimmsnarl_froslass_munkidori"
a08_root="$league/learners/a08_dipplin_seaking"

printf 'INSPECTION_STARTED=%s\n' "$(date -Iseconds)"

echo A02_LATEST_FAILURE
latest_failed=$(find "$a02_root" -mindepth 2 -maxdepth 2 -name FAILED.json -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
if [[ -n "$latest_failed" ]]; then
  printf 'PATH=%s\n' "$latest_failed"
  sed -n '1,240p' "$latest_failed"
  echo A02_LATEST_FAILURE_TRAIN_LOG
  tail -160 "$(dirname "$latest_failed")/train.log"
else
  echo NONE
fi

echo LEAGUE_STATE
"$python" -s - "$league/state/league.json" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text())
for chain in state.get("chains", []):
    if isinstance(chain, str):
        print(chain)
        continue
    keys = ("name", "generation", "failedUpdates", "publishedAgentDir", "checkpoint")
    print(json.dumps({key: chain.get(key) for key in keys}, ensure_ascii=False))
PY

echo LEARNER_PROCESSES
pgrep -af 'run_async_ppo_learner.py' || true

echo A08_PACKAGE
a08_agent=$(find "$a08_root" -type f -path '*/deployment/packages/*/main.py' -printf '%T@ %h\n' | sort -n | tail -1 | cut -d' ' -f2-)
if [[ -z "$a08_agent" ]]; then
  echo 'A08_AGENT_NOT_FOUND' >&2
  exit 20
fi
a08_generation=$(printf '%s\n' "$a08_agent" | sed -n 's#.*generation-\([0-9][0-9]*\)/.*#\1#p')
stamp=$(date -u +%Y%m%dT%H%M%SZ)
export_dir="$league/exports/current-a08-kaggle/a08-g${a08_generation}-${stamp}"
mkdir -p "$export_dir"
PYTHONNOUSERSITE=1 "$python" -s "$worktree/scripts/build_submission.py" \
  --agent "$a08_agent" \
  --cg-dir "$cg_dir" \
  --output "$export_dir/submission.tar.gz"
printf '%s\n' "$export_dir/submission.tar.gz" >"$league/exports/current-a08-kaggle/latest.txt"
printf 'A08_GENERATION=%s\n' "$a08_generation"
printf 'A08_AGENT=%s\n' "$a08_agent"
printf 'A08_ARCHIVE=%s\n' "$export_dir/submission.tar.gz"
stat -c 'A08_ARCHIVE_BYTES=%s' "$export_dir/submission.tar.gz"
tar -tzf "$export_dir/submission.tar.gz"

echo LOCAL_REPLAY_INDEX
if [[ -f /homes/lzhang/pocketmon/data/raw/replays/_index/manifest.csv ]]; then
  tail -12 /homes/lzhang/pocketmon/data/raw/replays/_index/manifest.csv
else
  echo MISSING_LOCAL_REPLAY_INDEX
fi

echo UNIVERSAL_BC_SOURCES
sources=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/final_7d/sources.json
if [[ -f "$sources" ]]; then
  "$python" -s - "$sources" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
payload = json.loads(p.read_text())
rows = payload.get("datasets", [payload.get("dataset")])
print("SOURCES_PATH=" + str(p))
print("SOURCES_MTIME=" + str(p.stat().st_mtime))
for row in rows:
    if row:
        print(json.dumps({key: row.get(key) for key in ("name", "decisions", "features", "tokenCache", "sequenceCache", "identityCache")}, ensure_ascii=False))
PY
else
  echo MISSING_UNIVERSAL_BC_SOURCES
fi

echo OFFICIAL_REPLAY_DATASETS
for day in 10 11 12; do
  ref="kaggle/pokemon-tcg-ai-battle-episodes-2026-08-${day}"
  printf 'DATASET=%s\n' "$ref"
  set +e
  PYTHONNOUSERSITE=1 "$python" -s -m kaggle datasets files "$ref" --csv 2>&1 | tail -12
  code=${PIPESTATUS[0]}
  set -e
  printf 'DATASET_EXIT=%s\n' "$code"
done

printf 'INSPECTION_FINISHED=%s\n' "$(date -Iseconds)"
