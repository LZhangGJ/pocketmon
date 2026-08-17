#!/usr/bin/env bash
set -Eeuo pipefail

: "${ROLE:?ROLE is required}"
: "${DECK:?DECK is required}"
: "${GPU_INDEX:?GPU_INDEX is required}"
: "${SEED_BASE:?SEED_BASE is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${WORKTREE:?WORKTREE is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${REFERENCE_ROOT:?REFERENCE_ROOT is required}"
: "${ENGINE_CATALOG:?ENGINE_CATALOG is required}"
: "${INITIAL_CHECKPOINT:?INITIAL_CHECKPOINT is required}"
: "${TEACHER_CHECKPOINT:?TEACHER_CHECKPOINT is required}"
: "${OPPONENT_POOL:?OPPONENT_POOL is required}"
: "${CG_DIR:?CG_DIR is required}"

PYTHON_BIN="${PYTHON_BIN:-/homes/lzhang/mypath/new/envs/trans/bin/python}"
GENERATIONS="${GENERATIONS:-20}"
EPISODES_PER_GENERATION="${EPISODES_PER_GENERATION:-200}"
SELF_PLAY_FRACTION="${SELF_PLAY_FRACTION:-0.25}"
TEMPERATURE="${TEMPERATURE:-1.0}"

actual_commit="$(git -C "$WORKTREE" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
  echo "commit mismatch: expected=$EXPECTED_COMMIT actual=$actual_commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$WORKTREE" status --porcelain)" ]]; then
  echo "worktree is not clean: $WORKTREE" >&2
  exit 3
fi
for required in \
  "$PYTHON_BIN" \
  "$DECK" \
  "$ENGINE_CATALOG" \
  "$INITIAL_CHECKPOINT" \
  "$TEACHER_CHECKPOINT" \
  "$OPPONENT_POOL"; do
  if [[ ! -e "$required" ]]; then
    echo "required input is missing: $required" >&2
    exit 4
  fi
done
if [[ ! -d "$REFERENCE_ROOT" || ! -d "$CG_DIR" ]]; then
  echo "reference root or official engine directory is missing" >&2
  exit 5
fi

mkdir -p "$RUN_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONNOUSERSITE=1
if [[ -f /homes/lzhang/mypath/new/envs/wsl4mis/lib/libstdc++.so.6 ]]; then
  export LD_PRELOAD=/homes/lzhang/mypath/new/envs/wsl4mis/lib/libstdc++.so.6
fi

current_checkpoint="$INITIAL_CHECKPOINT"
for generation in $(seq 1 "$GENERATIONS"); do
  generation_name="$(printf 'generation-%04d' "$generation")"
  generation_root="$RUN_ROOT/$generation_name"
  rollout="$generation_root/rollouts.jsonl.gz"
  checkpoint="$generation_root/checkpoint.pt"
  metrics="$generation_root/metrics.json"
  seed=$((SEED_BASE + generation))
  mkdir -p "$generation_root"

  if [[ -e "$checkpoint" || -e "$metrics" ]]; then
    if [[ ! -f "$checkpoint" || ! -f "$metrics" ]]; then
      echo "incomplete PPO training outputs: $generation_root" >&2
      exit 6
    fi
    echo "reusing completed PPO generation: $generation_root"
    current_checkpoint="$checkpoint"
    continue
  fi

  if [[ -f "$rollout" ]]; then
    echo "reusing completed PPO rollout: $rollout"
  else
    "$PYTHON_BIN" "$WORKTREE/experiment7/integration/collect_universal_ppo_rollouts.py" \
      --reference-root "$REFERENCE_ROOT" \
      --engine-catalog "$ENGINE_CATALOG" \
      --checkpoint "$current_checkpoint" \
      --teacher "$TEACHER_CHECKPOINT" \
      --deck "$DECK" \
      --pool "$OPPONENT_POOL" \
      --cg-dir "$CG_DIR" \
      --episodes "$EPISODES_PER_GENERATION" \
      --self-play-fraction "$SELF_PLAY_FRACTION" \
      --temperature "$TEMPERATURE" \
      --max-decisions 5000 \
      --seed "$seed" \
      --run-id "${ROLE}-${generation_name}" \
      --role "$ROLE" \
      --device cuda:0 \
      --output "$rollout" \
      >"$generation_root/collect.log" 2>&1
  fi

  "$PYTHON_BIN" "$WORKTREE/experiment7/integration/train_universal_ppo.py" \
    --reference-root "$REFERENCE_ROOT" \
    --rollouts "$rollout" \
    --initialize-from "$current_checkpoint" \
    --teacher "$TEACHER_CHECKPOINT" \
    --output "$checkpoint" \
    --metrics-output "$metrics" \
    --generation "$generation" \
    --role "$ROLE" \
    --seed "$seed" \
    --ppo-epochs 2 \
    --batch-size 128 \
    --learning-rate 1e-5 \
    --weight-decay 1e-5 \
    --gamma 0.997 \
    --gae-lambda 0.95 \
    --clip-ratio 0.1 \
    --value-clip 0.2 \
    --value-coefficient 0.5 \
    --entropy-coefficient 0.01 \
    --teacher-anchor-coefficient 0.02 \
    --gradient-clip-norm 0.5 \
    --target-kl 0.03 \
    --device cuda:0 \
    >"$generation_root/train.log" 2>&1

  current_checkpoint="$checkpoint"
done
