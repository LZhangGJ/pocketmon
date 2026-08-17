#!/usr/bin/env bash
set -euo pipefail

ssh -o BatchMode=yes -o ConnectTimeout=10 lzhang@10.113.13.78 \
  tar -C /homes/lzhang/mypath/new/envs/trans/lib/python3.11 \
    --exclude=site-packages -cf - . |
  tar -C /dev/shm/experiment7-ppo-python-env/lib/python3.11 -xf -

test -f /dev/shm/experiment7-ppo-python-env/lib/python3.11/xmlrpc/client.py
echo STDLIB_READY
