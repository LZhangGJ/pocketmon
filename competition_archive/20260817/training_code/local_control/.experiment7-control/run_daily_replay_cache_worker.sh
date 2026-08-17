#!/usr/bin/env bash
set -Eeuo pipefail

day=${1:?usage: $0 YYYY-MM-DD}
[[ "$day" =~ ^20[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]] || { echo "invalid date: $day" >&2; exit 2; }

worktree=/homes/lzhang/worktrees/experiment7-async-4c45f89
python=/homes/lzhang/mypath/new/envs/trans/bin/python
reference_root="$worktree/experiment7/reference"
engine_catalog=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/rebalance/2026-08-06-part-0-of-2-doraemon02/prepared/engine_catalog.json
root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc
raw_root=/dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays
worker_root="$root/workers/$day"
audit="$root/audits/$day.json"
prepared="$root/cache/$day/prepared"
tensor_store="$prepared/universal/features_tensordict"

snapshot() {
  local load cores cpu io
  load=$(cut -d' ' -f1 /proc/loadavg)
  cores=$(nproc --all)
  cpu=$(awk -v load_value="$load" -v cores="$cores" 'BEGIN {printf "%.2f",100*load_value/cores}')
  io=$(awk '/^some / {for(i=1;i<=NF;i++) if($i~/^avg10=/){split($i,a,"=");print a[2];exit}}' /proc/pressure/io 2>/dev/null || echo 0)
  printf '%s %s\n' "$cpu" "${io:-0}"
}

guarded() {
  local cpu io pid paused=0 status
  while true; do
    read -r cpu io < <(snapshot)
    awk -v cpu="$cpu" -v io="$io" 'BEGIN {exit !(cpu<95&&io<80)}' && break
    echo "RESOURCE_WAIT host=$(hostname) cpu=$cpu io=$io at=$(date -Iseconds)"
    sleep 30
  done
  setsid ionice -c2 -n7 nice -n10 "$@" & pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    read -r cpu io < <(snapshot)
    if awk -v cpu="$cpu" -v io="$io" 'BEGIN {exit !(cpu>=95||io>=80)}'; then
      if [[ $paused -eq 0 ]]; then kill -STOP -- "-$pid" 2>/dev/null || true; paused=1; fi
    elif [[ $paused -eq 1 ]] && awk -v cpu="$cpu" -v io="$io" 'BEGIN {exit !(cpu<85&&io<70)}'; then
      kill -CONT -- "-$pid" 2>/dev/null || true; paused=0
    fi
    sleep 30
  done
  [[ $paused -eq 0 ]] || kill -CONT -- "-$pid" 2>/dev/null || true
  set +e; wait "$pid"; status=$?; set -e
  return "$status"
}

run_worker() {
  mkdir -p "$worker_root" "$root/audits" "$root/logs" "$root/cache" "$raw_root"
  if [[ ! -s "$audit" ]]; then
    guarded env PYTHONNOUSERSITE=1 "$python" -s "$worktree/scripts/download_ptcg_data.py" \
      --date "$day" --max-episodes 0 --data-dir "$raw_root" --audit-output "$audit" \
      >"$root/logs/download-$day.log" 2>&1
  fi
  if [[ ! -s "$prepared/universal_training_sources.json" ]]; then
    mkdir -p "$prepared"
    guarded env PYTHONNOUSERSITE=1 "$python" -s "$worktree/experiment7/integration/prepare_universal_training_data.py" \
      --reference-root "$reference_root" --raw-root "$raw_root/$day" \
      --engine-catalog "$engine_catalog" --output-root "$prepared" --python "$python" \
      --policy-source winners --module-versions '*' --validation-fraction 0.05 \
      --strict-catalog --min-game-score-exclusive 900 \
      >"$root/logs/cache-$day.log" 2>&1
  fi
  test -s "$audit"
  test -s "$prepared/universal_training_sources.json"
  test -s "$tensor_store/meta.json"
  "$python" -s - "$prepared/universal_training_sources.json" "$tensor_store" <<'PY'
import json, pathlib, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
store = pathlib.Path(sys.argv[2]).resolve()
configured = pathlib.Path(manifest["dataset"]["features"]).resolve()
if configured != store:
    raise SystemExit(f"TENSOR_STORE_MANIFEST_MISMATCH configured={configured} expected={store}")
metadata = json.load(open(store / "meta.json", encoding="utf-8"))
if metadata.get("kind") != "experiment7_tensordict_memmap" or not metadata.get("tensors"):
    raise SystemExit(f"INVALID_TENSOR_STORE {store}")
print(f"TENSOR_STORE_READY path={store} tensors={len(metadata['tensors'])}")
PY
  guarded env PYTHONNOUSERSITE=1 PYTHONPATH="$worktree/experiment7/integration" \
    "$python" -s /homes/lzhang/build_daily_tensordict_window.py \
    >"$root/logs/tensordict-window-$day.log" 2>&1
  touch "$worker_root/SUCCESS"
  echo "DAILY_REPLAY_CACHE_SUCCESS day=$day host=$(hostname) at=$(date -Iseconds)"
}

if [[ "${2:-}" == "--run" ]]; then run_worker; exit; fi
mkdir -p "$worker_root"
pidfile="$worker_root/controller.pid"
if [[ -s "$pidfile" ]]; then
  old=$(<"$pidfile")
  if [[ "$old" =~ ^[0-9]+$ ]] && kill -0 "$old" 2>/dev/null; then
    echo "DAILY_REPLAY_CACHE_ALREADY_RUNNING day=$day pid=$old"; exit
  fi
fi
nohup /bin/bash "$0" "$day" --run >"$worker_root/controller.log" 2>&1 </dev/null &
pid=$!; printf '%s\n' "$pid" >"$pidfile"; sleep 2; kill -0 "$pid"
echo "DAILY_REPLAY_CACHE_STARTED day=$day pid=$pid host=$(hostname)"
