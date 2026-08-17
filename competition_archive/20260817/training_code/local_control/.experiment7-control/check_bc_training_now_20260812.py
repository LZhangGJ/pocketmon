#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path

root = Path('/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812')

def load(path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {'_error': str(exc)}

def tail(path, count=20):
    try:
        return path.read_text(errors='replace').replace('\x00','').splitlines()[-count:]
    except Exception:
        return []

out = {'checkedAt': time.time(), 'rolling': {}, 'capacity': {}}
for p in sorted((root / 'training').rglob('*')):
    if p.is_file() and p.name in ('training_report.json','summary.json','metrics.json','best_metrics.json'):
        out['rolling'][str(p.relative_to(root))] = load(p)
out['rolling']['controllerTail'] = tail(root / 'controller.log')

cap = root / 'capacity-comparison'
out['capacity']['controllerTail'] = tail(cap / 'controller.log')
exit_status = cap / 'control/exit-status.json'
out['capacity']['exitStatus'] = load(exit_status) if exit_status.exists() else None
for name in ('standard_1m','large_256x6'):
    node = {'logTail': tail(cap / 'logs' / f'{name}.log', 30), 'artifacts': []}
    candidate = cap / name
    if candidate.exists():
        for p in sorted(candidate.rglob('*')):
            if p.is_file() and p.name in ('training_report.json','summary.json','metrics.json','best_metrics.json','parity.json','smoke.json'):
                node['artifacts'].append({'path':str(p.relative_to(candidate)),'data':load(p),'mtime':p.stat().st_mtime})
        pts = sorted(candidate.rglob('*.pt'), key=lambda p:p.stat().st_mtime)
        node['checkpoints'] = len(pts)
        node['latestCheckpoint'] = str(pts[-1].relative_to(candidate)) if pts else None
        node['latestCheckpointMtime'] = pts[-1].stat().st_mtime if pts else None
    out['capacity'][name] = node
print(json.dumps(out, ensure_ascii=False))
