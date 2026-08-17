#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_ROOT WORKTREE EXPECTED_COMMIT" >&2
  exit 2
fi

run_root=$(realpath "$1")
worktree=$(realpath "$2")
expected_commit="$3"
launch_receipt="$run_root/continuation-g0021-g0040-launch.env"
controller="$run_root/../controllers/run_universal_ppo_role_edca178.sh"
resume_receipt="$run_root/resume-after-agent-isolation-${expected_commit:0:7}.env"

for required in "$launch_receipt" "$controller" "$worktree"; do
  if [[ ! -e "$required" ]]; then
    echo "required resume input is missing: $required" >&2
    exit 3
  fi
done

# The launch receipt was written with printf %q and is the authoritative source
# for the role-specific deck, GPU, seeds, checkpoints, and opponent pool.
# shellcheck disable=SC1090
source "$launch_receipt"
if [[ "$(realpath "$RUN_ROOT")" != "$run_root" ]]; then
  echo "launch receipt run root mismatch" >&2
  exit 4
fi
if [[ "$(git -C "$worktree" rev-parse HEAD)" != "$expected_commit" ]]; then
  echo "resume worktree commit mismatch" >&2
  exit 5
fi
if [[ -n "$(git -C "$worktree" status --porcelain)" ]]; then
  echo "resume worktree is not clean" >&2
  exit 6
fi
if [[ -e "$resume_receipt" ]]; then
  echo "refusing to overwrite resume receipt: $resume_receipt" >&2
  exit 7
fi

last_complete=0
for generation in $(seq 1 "$GENERATIONS"); do
  root="$run_root/$(printf 'generation-%04d' "$generation")"
  checkpoint="$root/checkpoint.pt"
  metrics="$root/metrics.json"
  if [[ -f "$checkpoint" && -f "$metrics" ]]; then
    last_complete="$generation"
  elif [[ -e "$checkpoint" || -e "$metrics" ]]; then
    echo "partial checkpoint/metrics pair requires manual review: $root" >&2
    exit 8
  else
    break
  fi
done
if (( last_complete < 20 )); then
  echo "resume requires at least a complete generation-0020" >&2
  exit 9
fi

WORKTREE="$worktree"
EXPECTED_COMMIT="$expected_commit"
{
  printf 'ROLE=%q\n' "$ROLE"
  printf 'RUN_ROOT=%q\n' "$run_root"
  printf 'WORKTREE=%q\n' "$WORKTREE"
  printf 'EXPECTED_COMMIT=%q\n' "$EXPECTED_COMMIT"
  printf 'LAST_COMPLETE_GENERATION=%q\n' "$last_complete"
  printf 'OPPONENT_POOL=%q\n' "$OPPONENT_POOL"
  printf 'OPPONENT_POOL_SHA256=%q\n' "$(sha256sum "$OPPONENT_POOL" | sed 's/ .*//')"
  printf 'RESUMED_AT_UTC=%q\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$resume_receipt"

export ROLE DECK GPU_INDEX SEED_BASE RUN_ROOT="$run_root" REFERENCE_ROOT ENGINE_CATALOG
export INITIAL_CHECKPOINT TEACHER_CHECKPOINT OPPONENT_POOL CG_DIR GENERATIONS PYTHON_BIN
export WORKTREE EXPECTED_COMMIT
exec bash "$controller"
