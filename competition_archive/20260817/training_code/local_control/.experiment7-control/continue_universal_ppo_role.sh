#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 14 ]]; then
  echo "usage: $0 ROLE DECK GPU SEED_BASE RUN_ROOT WORKTREE EXPECTED_COMMIT REFERENCE_ROOT ENGINE_CATALOG INITIAL_CHECKPOINT TEACHER_CHECKPOINT OPPONENT_POOL CG_DIR GENERATIONS" >&2
  exit 2
fi

role="$1"
deck=$(realpath "$2")
gpu="$3"
seed_base="$4"
run_root=$(realpath "$5")
worktree=$(realpath "$6")
expected_commit="$7"
reference_root=$(realpath "$8")
engine_catalog=$(realpath "$9")
initial_checkpoint=$(realpath "${10}")
teacher_checkpoint=$(realpath "${11}")
opponent_pool=$(realpath "${12}")
cg_dir=$(realpath "${13}")
generations="${14}"
python=/homes/lzhang/mypath/new/envs/trans/bin/python
controller="$run_root/../controllers/run_universal_ppo_role_edca178.sh"

case "$role" in
  generalist|hard_exploiter|diversity|conservative) ;;
  *) echo "unsupported PPO role: $role" >&2; exit 3 ;;
esac
if [[ ! "$gpu" =~ ^[0-9]+$ ]] || [[ ! "$seed_base" =~ ^[0-9]+$ ]]; then
  echo "GPU and seed base must be non-negative integers" >&2
  exit 4
fi
if [[ ! "$generations" =~ ^[0-9]+$ ]] || (( generations <= 20 )); then
  echo "continuation generation target must be greater than 20" >&2
  exit 5
fi
for required in \
  "$run_root/generation-0020/checkpoint.pt" "$run_root/generation-0020/metrics.json" \
  "$deck" "$engine_catalog" "$initial_checkpoint" "$teacher_checkpoint" \
  "$opponent_pool" "$controller"; do
  if [[ ! -e "$required" ]]; then
    echo "required continuation input is missing: $required" >&2
    exit 6
  fi
done
if [[ -e "$run_root/generation-0021" ]]; then
  echo "refusing to launch over an existing generation-0021: $run_root" >&2
  exit 7
fi

receipt="$run_root/continuation-g0021-g$(printf '%04d' "$generations")-launch.env"
if [[ -e "$receipt" ]]; then
  echo "refusing to overwrite continuation receipt: $receipt" >&2
  exit 8
fi
{
  printf 'ROLE=%q\n' "$role"
  printf 'DECK=%q\n' "$deck"
  printf 'GPU_INDEX=%q\n' "$gpu"
  printf 'SEED_BASE=%q\n' "$seed_base"
  printf 'RUN_ROOT=%q\n' "$run_root"
  printf 'WORKTREE=%q\n' "$worktree"
  printf 'EXPECTED_COMMIT=%q\n' "$expected_commit"
  printf 'REFERENCE_ROOT=%q\n' "$reference_root"
  printf 'ENGINE_CATALOG=%q\n' "$engine_catalog"
  printf 'INITIAL_CHECKPOINT=%q\n' "$initial_checkpoint"
  printf 'TEACHER_CHECKPOINT=%q\n' "$teacher_checkpoint"
  printf 'OPPONENT_POOL=%q\n' "$opponent_pool"
  printf 'CG_DIR=%q\n' "$cg_dir"
  printf 'GENERATIONS=%q\n' "$generations"
  printf 'PYTHON_BIN=%q\n' "$python"
  printf 'PARENT_G20_SHA256=%q\n' "$(sha256sum "$run_root/generation-0020/checkpoint.pt" | cut -d' ' -f1)"
  printf 'OPPONENT_POOL_SHA256=%q\n' "$(sha256sum "$opponent_pool" | cut -d' ' -f1)"
  printf 'LAUNCHED_AT_UTC=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$receipt"

export ROLE="$role"
export DECK="$deck"
export GPU_INDEX="$gpu"
export SEED_BASE="$seed_base"
export RUN_ROOT="$run_root"
export WORKTREE="$worktree"
export EXPECTED_COMMIT="$expected_commit"
export REFERENCE_ROOT="$reference_root"
export ENGINE_CATALOG="$engine_catalog"
export INITIAL_CHECKPOINT="$initial_checkpoint"
export TEACHER_CHECKPOINT="$teacher_checkpoint"
export OPPONENT_POOL="$opponent_pool"
export CG_DIR="$cg_dir"
export GENERATIONS="$generations"
export PYTHON_BIN="$python"
exec bash "$controller"
