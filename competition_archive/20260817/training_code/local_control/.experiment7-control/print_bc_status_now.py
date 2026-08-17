import json
from pathlib import Path

p=Path(__file__).parent/'bc-training-now.json'
d=json.loads(p.read_text(encoding='utf-8'))

def event_rows(lines):
    out=[]
    for line in lines:
        try:
            x=json.loads(line)
        except Exception:
            continue
        if isinstance(x,dict) and x.get('stage'):
            out.append({k:x.get(k) for k in ('stage','epoch','shard','policyNll','value','total','decisions','seconds','decisionsPerSecond')})
    return out

print('ROLLING_REPORTS', json.dumps(d['rolling'], ensure_ascii=True)[:8000])
for name in ('standard_1m','large_256x6'):
    row=d['capacity'][name]
    print(name, 'checkpoints',row.get('checkpoints'),'latest',row.get('latestCheckpoint'))
    print('ARTIFACTS',json.dumps(row.get('artifacts'),ensure_ascii=True)[:8000])
    print('EVENTS',json.dumps(event_rows(row.get('logTail',[])),ensure_ascii=True))
    print('TAIL_LAST',json.dumps(row.get('logTail',[])[-3:],ensure_ascii=True))
print('EXIT',d['capacity'].get('exitStatus'))
print('CONTROLLER',json.dumps(d['capacity'].get('controllerTail'),ensure_ascii=True))
