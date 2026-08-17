#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

root=Path('/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811')
raw=subprocess.check_output(['/homes/lzhang/mypath/new/envs/trans/bin/python','-s','/homes/lzhang/summarize_async_ppo_league.py','--league-root',str(root)],text=True)
summary=json.loads(raw)
league=json.loads((root/'state/league.json').read_text())
for name in ('a02_submission4_grimmsnarl_froslass_munkidori','a08_dipplin_seaking'):
    row=summary['chains'][name]
    batches=[]
    for p in (root/'learners'/name).glob('generation-*/batch.json'):
        try:
            d=json.loads(p.read_text())
            batches.append((p.stat().st_mtime,int(d.get('decisions') or 0),d.get('trainingControl'),str(p)))
        except Exception: pass
    batches.sort()
    recent=batches[-50:]
    controls=(league.get('chains',{}).get(name,{}) or {}).get('trainingControl')
    print(json.dumps({
        'chain':name,
        'generation':row.get('generation'),
        'completedShards':row.get('completedShards'),
        'episodes':row.get('episodes'),
        'decisions':row.get('decisions'),
        'decisionsPerEpisode':row.get('decisions')/row.get('episodes'),
        'decisionsPerGeneration':row.get('decisions')/max(1,row.get('generation')),
        'episodesPerGeneration':row.get('episodes')/max(1,row.get('generation')),
        'publishedUpdates':row.get('publishedUpdates'),
        'failedUpdates':row.get('failedUpdates'),
        'selfPlayEpisodes':row.get('selfPlayEpisodes'),
        'livePpoOpponentEpisodes':row.get('livePpoOpponentEpisodes'),
        'currentControl':controls,
        'batchFiles':len(batches),
        'recentBatchMeanDecisions':sum(x[1] for x in recent)/max(1,len(recent)),
        'recentBatchMinMax':[min((x[1] for x in recent),default=0),max((x[1] for x in recent),default=0)],
        'latestBatch':recent[-1][1:] if recent else None,
        'firstBatchMtime':batches[0][0] if batches else None,
        'lastBatchMtime':batches[-1][0] if batches else None,
    },ensure_ascii=False))
