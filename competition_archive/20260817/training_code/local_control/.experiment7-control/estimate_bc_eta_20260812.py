#!/usr/bin/env python3
import json
import os
import subprocess
import time
from pathlib import Path

root=Path('/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812')
cap=root/'capacity-comparison'
out={'now':time.time(),'models':{}}
for name in ('standard_1m','large_256x6'):
    log=cap/'logs'/f'{name}.log'
    events=[]
    if log.exists():
        for line in log.read_text(errors='replace').replace('\x00','').splitlines():
            try: row=json.loads(line)
            except Exception: continue
            if isinstance(row,dict) and row.get('stage'):
                events.append(row)
    by_epoch={}
    for row in events:
        epoch=row.get('epoch')
        if epoch is None: continue
        node=by_epoch.setdefault(str(epoch),{'train':[],'validation':[]})
        if row.get('stage')=='train_shard': node['train'].append(row)
        elif row.get('stage')=='validation_shard': node['validation'].append(row)
    report=cap/name/'training_report.json'
    checkpoints=sorted((cap/name/'checkpoints').glob('epoch_*.pt')) if (cap/name/'checkpoints').exists() else []
    out['models'][name]={
        'logMtime':log.stat().st_mtime if log.exists() else None,
        'lastEvent':events[-1] if events else None,
        'epochs':by_epoch,
        'report':json.loads(report.read_text()) if report.exists() else None,
        'checkpoints':[str(p) for p in checkpoints],
    }

manifest=json.loads((cap/'control/universal-10d-sources.json').read_text())
paths=[]
for row in manifest.get('datasets',[]):
    for key in ('features','tokenCache','sequenceCache','identityCache'):
        value=row.get(key)
        if value and value not in paths: paths.append(value)
sizes=[]
for path in paths:
    try:
        completed=subprocess.run(['du','-sb',path],capture_output=True,text=True,timeout=120,check=True)
        sizes.append((path,int(completed.stdout.split()[0])))
    except Exception as exc:
        sizes.append((path,None,type(exc).__name__))
out['cache']={'paths':len(paths),'totalBytes':sum(row[1] or 0 for row in sizes),'failed':sum(row[1] is None for row in sizes),'largest':sorted((row for row in sizes if row[1] is not None),key=lambda x:x[1],reverse=True)[:10]}
print(json.dumps(out,ensure_ascii=False))
