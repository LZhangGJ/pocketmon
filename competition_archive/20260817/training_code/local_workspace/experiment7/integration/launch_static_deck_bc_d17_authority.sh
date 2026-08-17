#!/usr/bin/env bash
set -euo pipefail

strict_root=${1:?strict root required}
control_root=${2:?control root required}
runtime=/dataT0/Free/lzhang/pocketmon-runs/experiment7-static-deck-bc-10d-20260815/control/runtime
expected_manifest_sha=b80edd05263002dbaf830dc4ad0296b7ede435f7a861461a9f28838a72d5010b

output=$(ssh doraemon17 \
  "EXPECTED_MANIFEST_SHA='$expected_manifest_sha' bash '$runtime/integration/launch_static_deck_bc_d17_builders.sh' '$strict_root'")
mkdir -p "$control_root/launch"
printf '%s\n' "$output" >"$control_root/launch/d17-authority-builders.tsv"
printf '%s\n' "$output"
