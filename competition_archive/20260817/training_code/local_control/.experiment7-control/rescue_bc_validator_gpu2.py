#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

RUNTIME = Path('/tmp/experiment7-async-bc-runtime-20260813')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, payload: dict) -> None:
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(payload, indent=2) + '\n')
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--previous-root', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--gpu', default='2')
    args = parser.parse_args()

    output = args.output_root.resolve() / args.profile
    state = output / 'rescue-validator-state.json'
    lock_file = (output / 'rescue-validator.lock').open('w')
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 0

    write(state, {'status': 'waiting_for_first_checkpoint', 'profile': args.profile, 'gpu': args.gpu, 'startedAt': now()})
    checkpoints = output / 'checkpoints'
    while not any(checkpoints.glob('epoch_*.pt')):
        controller_state = output / 'controller-state.json'
        if controller_state.exists():
            status = json.loads(controller_state.read_text()).get('status', '')
            if status.startswith('failed_before'):
                write(state, {'status': 'aborted_no_checkpoint', 'controllerStatus': status, 'endedAt': now()})
                return 1
        time.sleep(5)

    python = f'/proc/{os.getpid()}/exe'
    command = [
        'ionice', '-c2', '-n7', 'nice', '-n', '10', python, '-s',
        str(RUNTIME / 'validate_universal_bc_async.py'),
        '--sources', str(args.sources.resolve()), '--output-dir', str(output),
        '--baseline-report', str(args.previous_root.resolve() / 'training_report.json'),
        '--baseline-checkpoint', str(args.previous_root.resolve() / 'best_model.pt'),
        '--device', 'cuda:0', '--batch-size', '256', '--patience', '3',
        '--min-semantic-delta', '0.002', '--max-brier-increase', '0.005',
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (output / 'validation-gpu2-rescue.log').open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        write(state, {'status': 'validating', 'profile': args.profile, 'gpu': args.gpu, 'pid': process.pid, 'startedAt': now()})
        code = process.wait()
    write(state, {'status': 'complete' if code == 0 else 'failed', 'profile': args.profile, 'gpu': args.gpu, 'returnCode': code, 'endedAt': now()})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
