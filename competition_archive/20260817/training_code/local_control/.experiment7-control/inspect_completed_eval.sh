#!/usr/bin/env bash
set -u
round=/suedata1/Free/lzhang/pocketmon-runs/experiment7-hourly-cache-pool-distributed/monitoring/hourly-cache-pool/rounds/20260814T103053Z-allppo-11-2bf5d38f26e96c64
ls -ld "$round"
file=$(find "$round/raw" -name 'results-shard-*.csv' -type f | head -1)
echo "FILE=$file"
head -2 "$file"
grep -R -i -E 'invalid|fallback|entity.trunc|truncat' "$round/logs" 2>/dev/null | head -20 || true
