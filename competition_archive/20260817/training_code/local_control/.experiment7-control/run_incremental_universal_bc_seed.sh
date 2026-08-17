#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 SOURCES OUTPUT_ROOT INITIAL_CHECKPOINT WORKTREE EXPECTED_COMMIT SEED GPU_INDEX PYTHON" >&2
  exit 2
fi

sources=$(realpath "$1")
output_root="$2"
initial=$(realpath "$3")
worktree=$(realpath "$4")
expected_commit="$5"
seed="$6"
gpu_index="$7"
python=$(realpath "$8")

if [[ "$output_root" != /* ]] || [[ -e "$output_root" ]]; then
  echo "output root must be a new absolute path: $output_root" >&2
  exit 3
fi
if [[ ! "$seed" =~ ^[0-9]+$ || ! "$gpu_index" =~ ^[0-9]+$ ]]; then
  echo "seed and GPU index must be non-negative integers" >&2
  exit 4
fi
if [[ "$(git -C "$worktree" rev-parse HEAD)" != "$expected_commit" ]] || [[ -n "$(git -C "$worktree" status --porcelain)" ]]; then
  echo "incremental BC worktree is not the expected clean commit" >&2
  exit 5
fi

mkdir -p "$output_root"
{
  printf 'SOURCES=%q\n' "$sources"
  printf 'SOURCES_SHA256=%q\n' "$(sha256sum "$sources" | sed 's/ .*//')"
  printf 'OUTPUT_ROOT=%q\n' "$output_root"
  printf 'INITIAL_CHECKPOINT=%q\n' "$initial"
  printf 'INITIAL_CHECKPOINT_SHA256=%q\n' "$(sha256sum "$initial" | sed 's/ .*//')"
  printf 'WORKTREE=%q\n' "$worktree"
  printf 'EXPECTED_COMMIT=%q\n' "$expected_commit"
  printf 'SEED=%q\n' "$seed"
  printf 'GPU_INDEX=%q\n' "$gpu_index"
  printf 'EPOCHS=%q\n' 2
  printf 'LEARNING_RATE=%q\n' 5e-5
  printf 'LAUNCHED_AT_UTC=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$output_root/launch.env"

export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
if [[ -f /homes/lzhang/mypath/new/envs/wsl4mis/lib/libstdc++.so.6 ]]; then
  export LD_PRELOAD=/homes/lzhang/mypath/new/envs/wsl4mis/lib/libstdc++.so.6
fi
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
sha256sum "$output_root/launch.env" "$output_root/best_model.pt" \
  "$output_root/training_report.json" >"$output_root/SHA256SUMS"
touch "$output_root/SUCCESS"
echo "INCREMENTAL_UNIVERSAL_BC_SUCCESS seed=$seed output=$output_root"
