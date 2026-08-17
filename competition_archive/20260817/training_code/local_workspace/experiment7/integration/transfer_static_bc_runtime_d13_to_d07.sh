#!/usr/bin/env bash
set -euo pipefail

source_root=/homes/lzhang/mypath/new/envs/trans
target_host=doraemon07
target_root=/dev/shm/static-bc-trans-runtime
attempt=20260816T0135JST
incoming="${target_root}.incoming-${attempt}"

ssh "$target_host" "test ! -e '$target_root' && mkdir -p '$incoming'"
tar -C "$source_root" -cf - bin include lib share conda-meta \
  | ssh "$target_host" "tar -C '$incoming' -xf -"
ssh "$target_host" "test -x '$incoming/bin/python'; touch '$incoming/SUCCESS'; mv '$incoming' '$target_root'; du -sb '$target_root'"
