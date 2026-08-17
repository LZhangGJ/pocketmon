#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=${ROOT:-/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/capacity-comparison}
WORKTREE=${WORKTREE:-/homes/lzhang/worktrees/experiment7-async-4c45f89}
PYTHON_BIN=${PYTHON_BIN:-/homes/lzhang/mypath/new/envs/trans/bin/python}
CONTROL_ROOT=${CONTROL_ROOT:-/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-deck-branches-20260812/control}
GPU_STANDARD=${GPU_STANDARD:-1}
GPU_LARGE=${GPU_LARGE:-2}

old_root=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/daily
new_root=/dataT0/Free/lzhang/pocketmon-runs/replay-refresh-20260812/cache
manifests=(
  "$old_root/2026-08-02/prepared/universal_training_sources.json"
  "$old_root/2026-08-03/prepared/universal_training_sources.json"
  "$old_root/2026-08-04/prepared/universal_training_sources.json"
  "$old_root/2026-08-05/prepared/universal_training_sources.json"
  "$old_root/2026-08-06/prepared/universal_training_sources.json"
  "$old_root/2026-08-07/prepared/universal_training_sources.json"
  "$old_root/2026-08-08/prepared/universal_training_sources.json"
  "$new_root/2026-08-09/prepared/universal_training_sources.json"
  "$new_root/2026-08-10/prepared/universal_training_sources.json"
  "$new_root/2026-08-11/prepared/universal_training_sources.json"
)

mkdir -p "$ROOT/logs" "$ROOT/control"
echo "WAIT_CAPACITY_CACHES host=$(hostname) at=$(date -Iseconds)"
while true; do
  missing=0
  for manifest in "${manifests[@]}"; do
    [[ -s "$manifest" ]] || missing=$((missing + 1))
  done
  [[ $missing -eq 0 ]] && break
  echo "WAIT_CAPACITY_CACHES missing=$missing at=$(date -Iseconds)"
  sleep 60
done

sources="$ROOT/control/universal-10d-sources.json"
"$PYTHON_BIN" -s "$CONTROL_ROOT/build_universal_capacity_sources_20260812.py" \
  --output "$sources" "${manifests[@]}" | tee "$ROOT/control/source-build.json"

read -r load_value cores io < <(
  printf '%s ' "$(cut -d' ' -f1 /proc/loadavg)" "$(nproc)"
  if [[ -r /proc/pressure/io ]]; then
    awk '/^some / {for(i=1;i<=NF;i++) if($i ~ /^avg10=/){split($i,a,"="); print a[2]; exit}}' /proc/pressure/io
  else
    echo 0
  fi
)
cpu=$(awk -v load_value="$load_value" -v cores="$cores" 'BEGIN {printf "%.2f", 100*load_value/cores}')
if ! awk -v cpu="$cpu" -v io="$io" 'BEGIN {exit !(cpu < 95 && io < 80)}'; then
  echo "RESOURCE_GUARD_BLOCK cpu=$cpu io=$io at=$(date -Iseconds)" >&2
  exit 70
fi
echo "RESOURCE_GUARD_PASS cpu=$cpu io=$io at=$(date -Iseconds)"

train_one() {
  local name=$1 gpu=$2 d_model=$3 heads=$4 layers=$5 ff_dim=$6
  local out="$ROOT/$name"
  mkdir -p "$out"
  if [[ -s "$out/best_model.pt" && -s "$out/training_report.json" ]]; then
    echo "REUSE_TRAINED candidate=$name"
  else
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 \
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      "$PYTHON_BIN" -s "$WORKTREE/experiment7/integration/train_universal_bc.py" \
      --sources "$sources" --output-dir "$out" --device cuda:0 \
      --seed 20260812 --epochs 4 --batch-size 64 --learning-rate 2e-4 \
      --weight-decay 1e-4 --value-loss-weight 0.05 \
      --d-model "$d_model" --heads "$heads" --layers "$layers" \
      --ff-dim "$ff_dim" --dropout 0.05
  fi
  "$PYTHON_BIN" -s "$WORKTREE/experiment7/integration/export_and_package.py" export \
    --checkpoint "$out/best_model.pt" --output "$out/universal_bc.npz"
  "$PYTHON_BIN" -s "$WORKTREE/experiment7/integration/export_and_package.py" verify-universal \
    --reference-root "$WORKTREE/experiment7/reference" --sources "$sources" \
    --checkpoint "$out/best_model.pt" --portable "$out/universal_bc.npz" \
    --output "$out/parity.json" --python "$PYTHON_BIN" --decisions-per-source 150
  touch "$out/PARITY_PASSED"
}

set +e
train_one standard_1m "$GPU_STANDARD" 128 4 3 384 >"$ROOT/logs/standard_1m.log" 2>&1 &
standard_pid=$!
train_one large_256x6 "$GPU_LARGE" 256 8 6 1024 >"$ROOT/logs/large_256x6.log" 2>&1 &
large_pid=$!
printf '%s\n' "$standard_pid" >"$ROOT/control/standard.pid"
printf '%s\n' "$large_pid" >"$ROOT/control/large.pid"
wait "$standard_pid"; standard_status=$?
wait "$large_pid"; large_status=$?
set -e
printf '{"standard":%d,"large":%d}\n' "$standard_status" "$large_status" >"$ROOT/control/exit-status.json"
if [[ $standard_status -ne 0 || $large_status -ne 0 ]]; then
  touch "$ROOT/FAILED"
  exit 1
fi
touch "$ROOT/READY_FOR_SCREENING"
echo "CAPACITY_COMPARISON_READY at=$(date -Iseconds)"
