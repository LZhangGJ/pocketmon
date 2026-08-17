#!/usr/bin/env bash
set -u
hosts=(10.113.13.53 10.113.13.54 10.113.13.57 10.113.13.63 10.113.13.64 10.113.13.67 10.113.13.68 10.113.13.69 10.113.13.71 10.113.13.72 10.113.13.73 10.113.13.74 10.113.13.75 10.113.13.77 10.113.13.78)
tmp=$(mktemp -d)
for host in "${hosts[@]}"; do
  (
    result=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "bash --noprofile --norc -c 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null'" 2>/dev/null) || exit 0
    printf '%s\n' "$result" | grep -i A100 | sed "s/^/$host,/" > "$tmp/${host##*.}"
  ) &
done
wait
find "$tmp" -type f -size +0c -print0 | sort -z | xargs -0 -r cat
rm -rf "$tmp"
