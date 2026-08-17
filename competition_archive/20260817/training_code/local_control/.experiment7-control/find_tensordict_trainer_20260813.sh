#!/usr/bin/env bash
set -u
for trainer in /homes/lzhang/worktrees/*/experiment7/integration/train_universal_bc.py; do
  if grep -q 'features_tensordict\|tensordict-sources\|feature_tensor_store' "$trainer" 2>/dev/null; then
    printf '%s\n' "$trainer"
  fi
done
