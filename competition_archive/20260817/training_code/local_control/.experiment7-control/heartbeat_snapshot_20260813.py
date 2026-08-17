#!/usr/bin/env python3
import json, os, re
from pathlib import Path

def load(path):
    try: return json.loads(Path(path).read_text())
    except Exception as e: return {"_error": str(e), "_path": str(path)}

def metric(row):
    if not isinstance(row, dict): return None
    return row.get("scoreRate")

out={}
chall=Path('/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-checkpoint-challengers-20260812')
out['a08Candidates']={}
for g in (253,277,294,333):
    j=load(chall/f'g{g:06d}/monitoring/full-matrix/latest.json')
    c=j.get('chains',{}).get('a08_dipplin_seaking',{})
    agents={x.get('agent'):x for x in c.get('agents',[])}
    out['a08Candidates'][str(g)]={
      'status':j.get('status'),'games':j.get('games'),'failures':(c.get('frozenAggregate') or {}).get('failures'),
      'frozen':metric(c.get('frozenAggregate')),'bcFrozen':metric(c.get('universalBcFrozenAggregate')),
      'seat0':metric((c.get('seatMetrics') or {}).get('0')),'seat1':metric((c.get('seatMetrics') or {}).get('1')),
      'gap':c.get('seatGap'),'directBC':metric(c.get('directVsUniversalBc')),
      'arch':metric((agents.get('public_archaludon_meta') or {}).get('ppo')),
      'hardA06':metric((agents.get('hard_exploiter_g0010__05_a06_89e6155f2531') or {}).get('ppo')),
      'divA01':metric((agents.get('diversity_g0020__01_a01_ba51a134262b') or {}).get('ppo')),
    }

league=Path('/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811')
main=load(league/'monitoring/full-matrix/latest.json')
out['mainMatrix']={'status':main.get('status'),'updatedAt':main.get('updatedAt'),'games':main.get('games'),'agents':main.get('frozenAgentCount'),'roundId':main.get('roundId'),'chains':{}}
for name,c in main.get('chains',{}).items():
    out['mainMatrix']['chains'][name]={
      'generation':c.get('generation'),'frozen':metric(c.get('frozenAggregate')),
      'bc':metric(c.get('universalBcFrozenAggregate')),'delta':c.get('deltaVsPrevious'),
      'seat0':metric((c.get('seatMetrics') or {}).get('0')),'seat1':metric((c.get('seatMetrics') or {}).get('1')),
      'gap':c.get('seatGap'),'directBC':metric(c.get('directVsUniversalBc')),
    }

state=load(league/'state/adaptive-training-state.json')
out['adaptive']={'sourceRound':state.get('sourceRoundId') or state.get('lastAppliedRoundId'),'chains':state.get('chains',{})}

cap=Path('/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/capacity-comparison')
out['bc']={}
for name in ('standard_1m','large_256x6'):
    r=load(cap/name/'training_report.json')
    epochs=r.get('epochs',[])
    last=epochs[-1] if epochs else {}
    out['bc'][name]={
      'epochsCompleted':len(epochs),'bestEpoch':(r.get('best') or {}).get('epoch'),'bestSemantic':(r.get('best') or {}).get('score'),
      'lastSemantic':(last.get('validation') or {}).get('exactSemantic'),'lastDps':(last.get('training') or {}).get('decisionsPerSecond'),
      'parity':(cap/name/'PARITY_PASSED').exists(),'screenReady':(cap/'READY_FOR_SCREENING').exists(),
      'process':False,
    }
out['dailyRootExists']=Path('/dataT0/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc').exists()
print(json.dumps(out, ensure_ascii=False))
