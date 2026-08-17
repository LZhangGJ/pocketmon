import json
import sys
from pathlib import Path

d_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.experiment7-control/dist_latest_20260814.json')
o_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('.experiment7-control/old_latest_20260814.json')
d = json.loads(d_path.read_text(encoding='utf-8'))
o = json.loads(o_path.read_text(encoding='utf-8'))

print('ROUND', d['roundId'], d['games'], d['frozenAgentCount'], d['updatedAt'])
for name, c in d['chains'].items():
    a = c['frozenAggregate']
    s0, s1 = c['seatMetrics']['0'], c['seatMetrics']['1']
    old = o.get('chains', {}).get(name, {}).get('frozenAggregate', {})
    agents = sorted(c['agents'], key=lambda x: x['ppo']['scoreRate'] if x['ppo']['scoreRate'] is not None else -1)
    worst = ','.join(f"{x['agent']}={x['ppo']['scoreRate']:.0%}" for x in agents[:2])
    best = ','.join(f"{x['agent']}={x['ppo']['scoreRate']:.0%}" for x in agents[-2:])
    print('\t'.join([
        name, f"g{c['generation']}", f"{a['wins']}-{a['losses']}-{a['draws']}",
        f"{a['scoreRate']:.2%}", f"s0={s0['scoreRate']:.2%}", f"s1={s1['scoreRate']:.2%}",
        f"fail={a['failures']}", f"bcDelta={c['ppoMinusBc']:+.2%}",
        f"old={old.get('scoreRate', float('nan')):.2%}", f"worst:{worst}", f"best:{best}",
    ]))
