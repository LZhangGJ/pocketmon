#!/usr/bin/env bash
set -u
hosts=(10.113.13.53 10.113.13.54 10.113.13.57 10.113.13.63 10.113.13.64 10.113.13.67 10.113.13.68 10.113.13.69 10.113.13.71 10.113.13.72 10.113.13.73 10.113.13.74 10.113.13.75 10.113.13.77 10.113.13.78)
tmp=$(mktemp -d)
for host in "${hosts[@]}"; do
  (
    scp -q -o BatchMode=yes -o ConnectTimeout=7 /tmp/scan_project_memory_node_20260812.py "$host:/tmp/scan_project_memory_node_20260812.py" 2>/dev/null || exit 0
    ssh -o BatchMode=yes -o ConnectTimeout=7 "$host" "bash --noprofile --norc -c 'python3 /tmp/scan_project_memory_node_20260812.py'" > "$tmp/${host##*.}" 2>/dev/null || true
  ) &
done
wait
for host in "${hosts[@]}"; do f="$tmp/${host##*.}"; [ -s "$f" ] && echo "$host $(cat "$f")"; done
rm -rf "$tmp"
