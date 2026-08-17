#!/usr/bin/env bash
set -u
for host in 10.113.13.73 10.113.13.74 10.113.13.75 10.113.13.77; do
  echo "HOST=$host"
  ssh -o BatchMode=yes -o ConnectTimeout=6 "$host" "bash --noprofile --norc -c 'pgrep -af \"[t]rain_universal_bc.py|[r]un_universal_capacity|[p]ost_cache_bc\" || true'" || echo unreachable
done
