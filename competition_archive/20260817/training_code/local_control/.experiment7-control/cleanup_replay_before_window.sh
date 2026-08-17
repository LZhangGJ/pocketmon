#!/usr/bin/env bash
set -Eeuo pipefail

cutoff=${1:?usage: $0 EARLIEST-KEEP-DATE --confirmed-sources MANIFEST}
shift
[[ "$cutoff" =~ ^20[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]] || { echo "invalid cutoff: $cutoff" >&2; exit 2; }
[[ "${1:-}" == "--confirmed-sources" && -s "${2:-}" ]] || { echo "complete sources manifest required" >&2; exit 3; }

roots=(
  /dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays
  /dataT0/Free/lzhang/pocketmon-runs/replay-refresh-20260812/cache
  /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/daily
  /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-earlier-scoregt900-20260810/daily
  /dataT0/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/cache
)

targets=()
for root in "${roots[@]}"; do
  [[ -d "$root" ]] || continue
  root_resolved=$(realpath -e "$root")
  while IFS= read -r -d '' candidate; do
    name=$(basename "$candidate")
    [[ "$name" =~ ^20[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]] || continue
    [[ "$name" < "$cutoff" ]] || continue
    resolved=$(realpath -e "$candidate")
    case "$resolved" in "$root_resolved"/20??-??-??) ;; *) echo "REFUSE_TARGET=$resolved" >&2; exit 4;; esac
    targets+=("$resolved")
  done < <(find "$root_resolved" -mindepth 1 -maxdepth 1 -type d -print0)
done

for target in "${targets[@]}"; do du -sh -- "$target"; done
echo "DELETE_TARGET_COUNT=${#targets[@]} cutoff=$cutoff"
for target in "${targets[@]}"; do
  ionice -c3 nice -n19 rm -rf -- "$target"
  test ! -e "$target"
  echo "DELETE_DONE path=$target"
done
