#!/usr/bin/env bash
set -euo pipefail

src=/dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays/2026-08-06
base=/dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays/shards/experiment7-2026-08-06-scoregt900-rebalance
expected=/dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays/shards/experiment7-2026-08-06-scoregt900-rebalance

[[ "$(realpath -m "$base")" == "$expected" ]]
[[ -f "$src/manifest.csv" ]]
[[ ! -e "$base" ]]

mkdir -p "$base/part-0-of-2" "$base/part-1-of-2"
awk -F, 'NR == 1 || (($1 + 0) % 2) == 0' "$src/manifest.csv" > "$base/part-0-of-2/manifest.csv"
awk -F, 'NR == 1 || (($1 + 0) % 2) == 1' "$src/manifest.csv" > "$base/part-1-of-2/manifest.csv"

shopt -s nullglob
source_count=0
for replay in "$src"/*.json; do
    filename=${replay##*/}
    episode_id=${filename%.json}
    part=$((episode_id % 2))
    ln -s "$replay" "$base/part-$part-of-2/$filename"
    source_count=$((source_count + 1))
done

part0_count=$(find "$base/part-0-of-2" -maxdepth 1 -type l | wc -l)
part1_count=$(find "$base/part-1-of-2" -maxdepth 1 -type l | wc -l)
total_count=$(find "$base" -type l | wc -l)

printf 'PART0_JSON=%s\n' "$part0_count"
printf 'PART1_JSON=%s\n' "$part1_count"
printf 'TOTAL_LINKS=%s\n' "$total_count"
printf 'SOURCE_JSON=%s\n' "$source_count"
printf 'PART0_MANIFEST=%s\n' "$(wc -l < "$base/part-0-of-2/manifest.csv")"
printf 'PART1_MANIFEST=%s\n' "$(wc -l < "$base/part-1-of-2/manifest.csv")"
[[ "$total_count" -eq "$source_count" ]]
