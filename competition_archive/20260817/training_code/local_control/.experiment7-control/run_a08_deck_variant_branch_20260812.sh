#!/usr/bin/env bash
set -Eeuo pipefail

: "${BRANCH:?BRANCH is required}"
: "${DECK:?DECK is required}"
: "${GPU_INDEX:?GPU_INDEX is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${INITIAL_CHECKPOINT:?INITIAL_CHECKPOINT is required}"
: "${TARGET_POOL:?TARGET_POOL is required}"

WORKTREE=${WORKTREE:-/homes/lzhang/worktrees/experiment7-async-4c45f89}
# Rollout collection intentionally uses the adaptive worktree.  The formal PPO
# learner enforces a clean repository, so run it from an immutable worktree at
# the same base commit instead of rejecting required adaptive rollout changes.
LEARNER_WORKTREE=${LEARNER_WORKTREE:-/homes/lzhang/worktrees/experiment7-a08-learner-clean-8885a59}
PYTHON_BIN=${PYTHON_BIN:-/homes/lzhang/mypath/new/envs/trans/bin/python}
REFERENCE_ROOT=${REFERENCE_ROOT:-/homes/lzhang/pocketmon-worktrees/experiment7-248c61b/experiment7/reference}
ENGINE_CATALOG=${ENGINE_CATALOG:-/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/rebalance/2026-08-06-part-0-of-2-doraemon02/prepared/engine_catalog.json}
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:-/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/final_7d/training/seed-20260812-doraemon16-gpu0/best_model.pt}
MAIN_POOL=${MAIN_POOL:-/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/state/opponent-pool-live.json}
CG_DIR=${CG_DIR:-/dataT0/Free/lzhang/pocketmon-runs/experiment7-opponent-pool-20260810/official-engine/cg}
GENERATIONS=${GENERATIONS:-20}
SEED_BASE=${SEED_BASE:-2026081200}

resource_snapshot() {
  local cores load_value cpu io
  # Use the machine-wide CPU count.  `nproc` without --all can reflect the
  # controller's affinity/cgroup mask and made an otherwise idle 80-core host
  # look >200% loaded, permanently blocking rollout collection.
  cores=$(nproc --all)
  load_value=$(awk '{print $1}' /proc/loadavg)
  cpu=$(awk -v load_value="$load_value" -v cores="$cores" 'BEGIN { printf "%.2f", 100.0 * load_value / cores }')
  io=0
  if [[ -r /proc/pressure/io ]]; then
    io=$(awk '/^some / {for(i=1;i<=NF;i++) if($i ~ /^avg10=/){split($i,a,"="); print a[2]; exit}}' /proc/pressure/io)
  fi
  printf '%s %s\n' "$cpu" "${io:-0}"
}

wait_for_capacity() {
  while true; do
    read -r cpu io < <(resource_snapshot)
    if awk -v cpu="$cpu" -v io="$io" 'BEGIN { exit !(cpu < 95 && io < 80) }'; then
      printf 'LOAD_GUARD_PASS branch=%s host=%s cpu=%s io=%s at=%s\n' \
        "$BRANCH" "$(hostname)" "$cpu" "$io" "$(date -Iseconds)"
      return
    fi
    printf 'LOAD_GUARD_WAIT branch=%s host=%s cpu=%s io=%s at=%s\n' \
      "$BRANCH" "$(hostname)" "$cpu" "$io" "$(date -Iseconds)"
    sleep 30
  done
}

for required in "$PYTHON_BIN" "$DECK" "$INITIAL_CHECKPOINT" "$TEACHER_CHECKPOINT" \
  "$MAIN_POOL" "$TARGET_POOL" "$ENGINE_CATALOG"; do
  [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 2; }
done
[[ -d "$REFERENCE_ROOT" && -d "$CG_DIR" && -d "$WORKTREE" && -d "$LEARNER_WORKTREE" ]] || {
  echo "missing reference/cg/worktree directory" >&2
  exit 3
}
[[ $(wc -l <"$DECK") -eq 60 ]] || { echo "deck must have 60 lines: $DECK" >&2; exit 4; }

mkdir -p "$RUN_ROOT/$BRANCH"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
if [[ -f /homes/lzhang/mypath/new/envs/wsl4mis/lib/libstdc++.so.6 ]]; then
  export LD_PRELOAD=/homes/lzhang/mypath/new/envs/wsl4mis/lib/libstdc++.so.6
fi

current_checkpoint="$INITIAL_CHECKPOINT"
for generation in $(seq 1 "$GENERATIONS"); do
  generation_name=$(printf 'generation-%04d' "$generation")
  generation_root="$RUN_ROOT/$BRANCH/$generation_name"
  checkpoint="$generation_root/checkpoint.pt"
  metrics="$generation_root/metrics.json"
  main_rollout="$generation_root/main-frozen90.jsonl.gz"
  self_rollout="$generation_root/selfplay50.jsonl.gz"
  target_rollout="$generation_root/targeted60.jsonl.gz"
  seed=$((SEED_BASE + generation * 10))
  mkdir -p "$generation_root"

  if [[ -s "$checkpoint" && -s "$metrics" ]]; then
    current_checkpoint="$checkpoint"
    echo "REUSE_GENERATION branch=$BRANCH generation=$generation"
    continue
  fi
  if [[ -e "$checkpoint" || -e "$metrics" ]]; then
    echo "incomplete generation outputs: $generation_root" >&2
    exit 5
  fi

  wait_for_capacity
  if [[ ! -s "$main_rollout" ]]; then
    ionice -c 2 -n 7 nice -n 10 "$PYTHON_BIN" -s \
      "$WORKTREE/experiment7/integration/collect_universal_ppo_rollouts.py" \
      --reference-root "$REFERENCE_ROOT" --engine-catalog "$ENGINE_CATALOG" \
      --checkpoint "$current_checkpoint" --teacher "$TEACHER_CHECKPOINT" \
      --deck "$DECK" --pool "$MAIN_POOL" --cg-dir "$CG_DIR" \
      --episodes 90 --self-play-fraction 0 --temperature 1.0 --max-decisions 5000 \
      --seed "$((seed + 1))" --run-id "$BRANCH-$generation_name-main" \
      --behavior-generation "$((generation - 1))" \
      --behavior-snapshot-id "$BRANCH-g$(printf '%04d' $((generation - 1)))" \
      --role diversity --device cuda:0 --output "$main_rollout" \
      >"$generation_root/collect-main.log" 2>&1
  fi

  wait_for_capacity
  if [[ ! -s "$self_rollout" ]]; then
    ionice -c 2 -n 7 nice -n 10 "$PYTHON_BIN" -s \
      "$WORKTREE/experiment7/integration/collect_universal_ppo_rollouts.py" \
      --reference-root "$REFERENCE_ROOT" --engine-catalog "$ENGINE_CATALOG" \
      --checkpoint "$current_checkpoint" --teacher "$TEACHER_CHECKPOINT" \
      --deck "$DECK" --pool "$MAIN_POOL" --cg-dir "$CG_DIR" \
      --episodes 50 --self-play-fraction 1 --temperature 1.0 --max-decisions 5000 \
      --seed "$((seed + 2))" --run-id "$BRANCH-$generation_name-self" \
      --behavior-generation "$((generation - 1))" \
      --behavior-snapshot-id "$BRANCH-g$(printf '%04d' $((generation - 1)))" \
      --role diversity --device cuda:0 --output "$self_rollout" \
      >"$generation_root/collect-self.log" 2>&1
  fi

  wait_for_capacity
  if [[ ! -s "$target_rollout" ]]; then
    ionice -c 2 -n 7 nice -n 10 "$PYTHON_BIN" -s \
      "$WORKTREE/experiment7/integration/collect_universal_ppo_rollouts.py" \
      --reference-root "$REFERENCE_ROOT" --engine-catalog "$ENGINE_CATALOG" \
      --checkpoint "$current_checkpoint" --teacher "$TEACHER_CHECKPOINT" \
      --deck "$DECK" --pool "$TARGET_POOL" --cg-dir "$CG_DIR" \
      --episodes 60 --self-play-fraction 0 --temperature 1.0 --max-decisions 5000 \
      --seed "$((seed + 3))" --run-id "$BRANCH-$generation_name-target" \
      --behavior-generation "$((generation - 1))" \
      --behavior-snapshot-id "$BRANCH-g$(printf '%04d' $((generation - 1)))" \
      --role generalist --device cuda:0 --output "$target_rollout" \
      >"$generation_root/collect-target.log" 2>&1
  fi

  wait_for_capacity
  "$PYTHON_BIN" -s "$LEARNER_WORKTREE/experiment7/integration/train_universal_ppo.py" \
    --reference-root "$REFERENCE_ROOT" \
    --rollouts "$main_rollout" "$self_rollout" "$target_rollout" \
    --initialize-from "$current_checkpoint" --teacher "$TEACHER_CHECKPOINT" \
    --output "$checkpoint" --metrics-output "$metrics" \
    --generation "$generation" --role diversity --seed "$((seed + 4))" \
    --ppo-epochs 1 --batch-size 128 --learning-rate 5e-6 --weight-decay 1e-5 \
    --gamma 0.997 --gae-lambda 0.95 --clip-ratio 0.1 --value-clip 0.2 \
    --value-coefficient 0.5 --entropy-coefficient 0.01 \
    --teacher-anchor-coefficient 0.04 --seat1-weight 2.0 \
    --normalize-advantages-by-player --balance-player-minibatches \
    --gradient-clip-norm 0.5 --target-kl 0.03 --device cuda:0 \
    >"$generation_root/train.log" 2>&1

  current_checkpoint="$checkpoint"
  printf 'BRANCH_GENERATION_COMPLETE branch=%s generation=%s checkpoint=%s at=%s\n' \
    "$BRANCH" "$generation" "$checkpoint" "$(date -Iseconds)"
  if [[ "$generation" -eq 10 || "$generation" -eq 20 ]]; then
    printf '%s\n' "$checkpoint" >"$generation_root/READY_FOR_EVAL"
  fi
done

echo "BRANCH_COMPLETE branch=$BRANCH generations=$GENERATIONS at=$(date -Iseconds)"
