#!/usr/bin/env bash
set -u
for host in 10.113.13.74 10.113.13.75 10.113.13.73 10.113.13.78; do
  echo "HOST=$host"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "bash --noprofile --norc -c 'pgrep -af \"[r]un_async_ppo_learner.py|[t]rain_universal_ppo.py|[t]rain_universal_bc.py.*capacity-comparison-a100\" || true'" || echo unreachable
done
echo A100_LOG
ssh -o BatchMode=yes -o ConnectTimeout=8 10.113.13.78 "bash --noprofile --norc -c 'tail -50 /tmp/launch_a100_capacity_fasttrack_20260812.log 2>/dev/null || true; pgrep -af \"[l]aunch_a100_capacity_fasttrack|[r]sync.*lzhang-bc-capacity\" || true'" || true
