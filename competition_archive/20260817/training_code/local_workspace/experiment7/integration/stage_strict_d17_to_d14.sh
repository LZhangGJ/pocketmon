#!/usr/bin/env bash
set -euo pipefail

source_root=${1:-/tmp/lzhang-strict-scoregt1000-window-20260815T1200Z}
target_host=${2:-doraemon14}
target_root=${3:-/tmp/lzhang-strict-scoregt1000-window-20260815T1200Z}
expected_sha=${EXPECTED_MANIFEST_SHA:-b80edd05263002dbaf830dc4ad0296b7ede435f7a861461a9f28838a72d5010b}
control=${4:-/tmp/lzhang-strict-scoregt1000-window-20260815T1200Z-transfer}
mkdir -p "$control"

test -f "$source_root/SUCCESS"
actual_sha=$(sha256sum "$source_root/tensordict-sources.json" | awk '{print $1}')
test "$actual_sha" = "$expected_sha"
ssh "$target_host" "test ! -f '$target_root/SUCCESS' || test \"\$(sha256sum '$target_root/tensordict-sources.json' | awk '{print \$1}')\" = '$expected_sha'"

if [[ -f "$control/rsync.pid" ]] && kill -0 "$(cat "$control/rsync.pid")" 2>/dev/null; then
  cat "$control/rsync.pid"
  exit 0
fi
nohup rsync -a --sparse --info=progress2 \
  "$source_root/" "$target_host:$target_root/" \
  >"$control/rsync.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$control/rsync.pid"
printf '%s\n' "$pid"
