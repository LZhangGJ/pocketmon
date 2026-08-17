#!/usr/bin/env bash
set -euo pipefail

source_root=/homes/lzhang/mypath/new/envs/trans
target_host=doraemon07
target_root=/dev/shm/static-bc-trans-min
incoming="${target_root}.incoming-20260816T0147JST"

ssh "$target_host" "test ! -e '$target_root' && mkdir -p '$incoming'"
tar --exclude='lib/python3.11/site-packages' -C "$source_root" -cf - \
  bin/python bin/python3 bin/python3.11 lib/python3.11 \
  | ssh "$target_host" "tar -C '$incoming' -xf -"
tar -C "$source_root" -cf - \
  lib/python3.11/site-packages/filelock* \
  lib/python3.11/site-packages/fsspec* \
  lib/python3.11/site-packages/functorch \
  lib/python3.11/site-packages/isympy.py \
  lib/python3.11/site-packages/jinja2* \
  lib/python3.11/site-packages/markupsafe* \
  lib/python3.11/site-packages/mpmath* \
  lib/python3.11/site-packages/networkx* \
  lib/python3.11/site-packages/numpy* \
  lib/python3.11/site-packages/nvidia* \
  lib/python3.11/site-packages/sympy* \
  lib/python3.11/site-packages/torch* \
  lib/python3.11/site-packages/triton* \
  lib/python3.11/site-packages/typing_extensions* \
  | ssh "$target_host" "tar -C '$incoming' -xf -"
ssh "$target_host" "test -x '$incoming/bin/python'; touch '$incoming/SUCCESS'; mv '$incoming' '$target_root'; du -sb '$target_root'"
