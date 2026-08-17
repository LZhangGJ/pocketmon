#!/usr/bin/env bash
set -u
pgrep -af 'launch_a100_capacity_fasttrack|rsync.*lzhang-bc-capacity|train_universal_bc.py.*capacity-comparison-a100' || true
du -sh /tmp/lzhang-bc-capacity-a100-20260812 2>/dev/null || true
ls -l /dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/capacity-comparison-a100-b256 2>/dev/null || true
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | sed -n '2p;4p'
