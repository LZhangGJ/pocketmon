#!/usr/bin/env bash
set -euo pipefail

source_host=${1:-10.113.13.78}
source_root=${2:-/dev/shm/experiment7-large-g1-recovery-Vr8UhfPZ/runtime/python-env}
target_root=${3:-/dev/shm/experiment7-ppo-python-env}

if [[ -e "$target_root/COPYING" ]]; then
  echo "copy already active: $target_root" >&2
  exit 3
fi
mkdir -p "$target_root"
touch "$target_root/COPYING"
trap 'rm -f "$target_root/COPYING"' EXIT

ssh -o BatchMode=yes -o ConnectTimeout=10 "lzhang@$source_host" \
  tar -C "$source_root" -cf - . | tar -C "$target_root" -xf -

"$target_root/bin/python" -c 'import torch; assert torch.cuda.is_available()'
touch "$target_root/SUCCESS"
echo "RUNTIME_READY $target_root"
