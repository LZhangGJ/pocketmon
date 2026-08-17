#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 SOURCES OUTPUT_ROOT INITIAL_CHECKPOINT WORKTREE SEED GPU_INDEX" >&2
  exit 2
fi
sources=$(realpath "$1")
output_root="$2"
initial=$(realpath "$3")
worktree=$(realpath "$4")
seed="$5"
gpu_index="$6"
python=/homes/lzhang/mypath/new/envs/trans/bin/python

if [[ "$output_root" != /* ]] || [[ -e "$output_root" ]]; then
  echo "output root must be a new absolute path: $output_root" >&2
  exit 3
fi
mkdir -p "$output_root"
{
  printf 'SOURCES=%q\n' "$sources"
  printf 'OUTPUT_ROOT=%q\n' "$output_root"
  printf 'INITIAL_CHECKPOINT=%q\n' "$initial"
  printf 'WORKTREE=%q\n' "$worktree"
  printf 'WORKTREE_COMMIT=%q\n' "$(git -C "$worktree" rev-parse HEAD)"
  printf 'SEED=%q\n' "$seed"
  printf 'GPU_INDEX=%q\n' "$gpu_index"
  printf 'HASH_VALIDATION=%q\n' omitted
  printf 'LAUNCHED_AT_UTC=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$output_root/launch.env"

export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_PRELOAD=/homes/lzhang/mypath/new/envs/wsl4mis/lib/libstdc++.so.6
cd "$worktree"
"$python" -s "$worktree/experiment7/integration/train_universal_bc.py" \
  --sources "$sources" \
  --output-dir "$output_root" \
  --initialize-from "$initial" \
  --device cuda:0 \
  --seed "$seed" \
  --epochs 2 \
  --batch-size 64 \
  --learning-rate 5e-5 \
  --weight-decay 1e-4 \
  --value-loss-weight 0.05 \
  --d-model 128 \
  --heads 4 \
  --layers 3 \
  --ff-dim 384 \
  --dropout 0.05 \
  >"$output_root/train.log" 2>&1
test -f "$output_root/best_model.pt"
test -f "$output_root/training_report.json"
touch "$output_root/SUCCESS"
echo "INCREMENTAL_UNIVERSAL_BC_SUCCESS seed=$seed output=$output_root"
