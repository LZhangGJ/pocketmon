import json
from pathlib import Path

base = Path(__file__).parent
h = json.loads((base / 'heartbeat-1136.json').read_text(encoding='utf-8'))
d = json.loads((base / 'heartbeat-dynamic-1136.json').read_text(encoding='utf-8'))

print('FULL_META', {k:h['full'].get(k) for k in ('status','updatedAt','roundId','games','frozenAgentCount')})
for name,row in h['training']['chains'].items():
    print('TRAIN', name, {k:row.get(k) for k in ('generation','completedShards','episodes','decisions','externalWins','externalLosses','selfPlayEpisodes','livePpoOpponentEpisodes','publishedUpdates','failedUpdates','latestInitialPolicyShift','latestEpoch')})
for name,row in h['full']['chains'].items():
    print('FULL', name, {k:row.get(k) for k in ('generation','frozenAggregate','universalBcFrozenAggregate','ppoMinusBc','deltaVsPrevious','progress','seatMetrics','seatGap','directVsUniversalBc','ppoHeadToHead')})
chain_short = {
    'a02_submission4_grimmsnarl_froslass_munkidori':'A02',
    'a05_raging_bolt_ogerpon_kangaskhan':'A05',
    'a08_dipplin_seaking':'A08',
    'mega_lucario_ex':'Luc',
}
agents = {}
for chain,row in h['full']['chains'].items():
    for agent in row.get('agents',[]):
        agents.setdefault(agent[0], {})[chain_short[chain]] = agent[1:]
print('AGENT_TABLE_START')
for agent, rows in agents.items():
    cells=[]
    for short in ('A02','A05','A08','Luc'):
        v=rows.get(short)
        cells.append('-' if not v else '/'.join(f'{x*100:.0f}' for x in v))
    print('|'+agent+'|'+'|'.join(cells)+'|')
print('AGENT_TABLE_END')
print('SUBMISSION4', h.get('submission4'))
print('TARGETED', h.get('targetedSummary'))
print('VARIANTS', h.get('a08Variants'))
replay = h.get('replay') or {}
print('REPLAY', {'cacheManifests': replay.get('cacheManifests'), 'auditKeys': sorted((replay.get('audits') or {}).keys())})
inc = h.get('incremental') or {}
print('INCREMENTAL_KEYS', sorted(inc.keys()))
for key in ('standard_1m','large_256x6','capacity-comparison/control/exit-status.json'):
    if key in inc:
        print('INCREMENTAL_ITEM', key, json.dumps(inc[key], ensure_ascii=True)[:4000])
for key in ('standard_1m.log','large_256x6.log','capacity-comparison/controller.log'):
    if key in inc:
        print('INCREMENTAL_LOG', key, json.dumps(inc[key], ensure_ascii=True)[-3000:])
print('DYNAMIC_META', {k:d.get(k) for k in ('leagueUpdatedAt','sourceRoundId','state')})
for name,row in d.get('chains',{}).items():
    control=row.get('control') or {}
    rollout=control.get('rollout') or {}
    learner=control.get('learner') or {}
    print('DYNAMIC', name, {
        'generation':row.get('generation'),
        'evidence':control.get('evidence'),
        'rollout':{k:rollout.get(k) for k in ('selfPlayFraction','learnerSeat1Fraction','archetypeWeights')},
        'weakAgents': sorted((rollout.get('agentWeights') or {}).items(), key=lambda x:(-x[1],x[0]))[:5],
        'learner':learner,
        'latestSamplingControl':row.get('latestSamplingControl'),
        'latestSummaryDecisions':row.get('latestSummaryDecisions'),
        'latestBatchControl':row.get('latestBatchControl'),
        'latestBatchDecisions':row.get('latestBatchDecisions'),
    })
