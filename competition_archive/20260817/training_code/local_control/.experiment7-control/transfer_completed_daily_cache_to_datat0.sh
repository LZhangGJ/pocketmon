#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 DD" >&2
    exit 2
fi

day=$1
case "$day" in
    02|03|04|05|08) ;;
    *) echo "unsupported day: $day" >&2; exit 2 ;;
esac

source_root="/homes/lzhang/pocketmon/runs/experiment7-universal-7d-scoregt900-20260810/daily/2026-08-$day"
destination_parent=/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/daily
destination_root="$destination_parent/2026-08-$day"
temporary_root="$destination_parent/.transfer-2026-08-$day-partial"

[[ "$(realpath -m "$source_root")" == "/homes/lzhang/pocketmon/runs/experiment7-universal-7d-scoregt900-20260810/daily/2026-08-$day" ]]
[[ "$(realpath -m "$destination_root")" == "/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/daily/2026-08-$day" ]]
[[ -f "$source_root/prepared/universal_training_sources.json" ]]
[[ ! -e "$destination_root" ]]
[[ ! -e "$temporary_root" ]]
command -v rsync >/dev/null

mkdir -p "$temporary_root"
rsync -a --info=progress2 "$source_root/" "$temporary_root/"

verification=$(rsync -a --checksum --dry-run --itemize-changes "$source_root/" "$temporary_root/")
if [[ -n "$verification" ]]; then
    printf '%s\n' "$verification" >&2
    echo "checksum verification reported differences" >&2
    exit 3
fi

source_manifest_sha=$(sha256sum "$source_root/prepared/universal_training_sources.json" | awk '{print $1}')
destination_manifest_sha=$(sha256sum "$temporary_root/prepared/universal_training_sources.json" | awk '{print $1}')
[[ "$source_manifest_sha" == "$destination_manifest_sha" ]]

source_files=$(find "$source_root" -type f | wc -l)
destination_files=$(find "$temporary_root" -type f | wc -l)
[[ "$source_files" -eq "$destination_files" ]]

mv "$temporary_root" "$destination_root"
printf 'TRANSFER_COMPLETE day=2026-08-%s files=%s manifest_sha256=%s destination=%s\n' \
    "$day" "$source_files" "$source_manifest_sha" "$destination_root"
