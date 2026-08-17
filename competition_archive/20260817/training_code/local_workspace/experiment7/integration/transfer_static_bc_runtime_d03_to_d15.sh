#!/usr/bin/env bash
set -euo pipefail

source_root=${SOURCE_ROOT:-/homes/lzhang/mypath/new/envs/trans}
target_host=${TARGET_HOST:-lzhang@10.113.13.73}
target_root=${TARGET_ROOT:-/dev/shm/static-bc-trans-runtime}
attempt=${ATTEMPT_ID:-static-bc-trans-runtime-20260816T0038JST}
incoming="${target_root}.incoming-${attempt}"
progress=${PROGRESS_LOG:-/tmp/${attempt}.progress.log}

[[ "$(realpath -m "$source_root")" == /homes/lzhang/mypath/new/envs/trans ]]
[[ "$target_root" == /dev/shm/static-bc-trans-runtime ]]
[[ -x "$source_root/bin/python3.11" ]]

ssh -o BatchMode=yes "$target_host" \
  "test ! -e '$target_root'; test ! -e '$incoming'; mkdir -p '$incoming'; echo \$\$ > /tmp/${attempt}.destination.pid"

site_packages=(
  lib/python3.11/site-packages/numpy
  lib/python3.11/site-packages/numpy.libs
  lib/python3.11/site-packages/nvidia
  lib/python3.11/site-packages/torch
)
for optional in \
  typing_extensions.py filelock fsspec jinja2 markupsafe networkx sympy mpmath; do
  candidate="lib/python3.11/site-packages/$optional"
  [[ ! -e "$source_root/$candidate" ]] || site_packages+=("$candidate")
done

runtime_files=(bin/python bin/python3 bin/python3.11 lib/python3.11)
for pattern in \
  libbz2.so* libcrypto.so* libffi.so* libgcc_s.so* \
  liblzma.so* libssl.so* libuuid.so* libz.so*; do
  for path in "$source_root"/lib/$pattern; do
    [[ ! -e "$path" && ! -L "$path" ]] || runtime_files+=("${path#"$source_root"/}")
  done
done

{
  tar --sparse \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='*/tests' \
    --exclude='*/test' \
    --exclude='lib/python3.11/site-packages/*' \
    -C "$source_root" -cf - "${runtime_files[@]}"
  tar --sparse \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='*/tests' \
    --exclude='*/test' \
    -C "$source_root" -cf - "${site_packages[@]}"
} \
  | dd bs=4M status=progress 2>"$progress" \
  | ssh -o BatchMode=yes "$target_host" \
      "set -eu; tar --ignore-zeros -C '$incoming' -xf -; test -x '$incoming/bin/python3.11'; mv '$incoming' '$target_root'; printf 'SUCCESS\n' > '${target_root}.SUCCESS'"

ssh -o BatchMode=yes "$target_host" \
  "du -sb '$target_root'; readlink -f '$target_root/bin/python'; test -f '${target_root}.SUCCESS'"
