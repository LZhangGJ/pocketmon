import json
from collections import defaultdict
from pathlib import Path

files = [
    Path('.experiment7-control/dist_latest_20260814.json'),
    Path('.experiment7-control/dist_latest_20260814_113809.json'),
    Path('.experiment7-control/dist_latest_20260814_123520.json'),
    Path('.experiment7-control/dist_latest_20260814_133525.json'),
]
rows = defaultdict(list)
for path in files:
    p = json.loads(path.read_text(encoding='utf-8'))
    c = p['chains']['dragapult_munkidori_large_g9']
    gen = c['generation']
    print('GEN', gen, 'HEAD_TO_HEAD')
    for name, m in sorted(c['ppoHeadToHead'].items(), key=lambda kv: kv[1]['scoreRate'] if kv[1]['scoreRate'] is not None else -1):
        print(name, m['wins'], m['losses'], m['draws'], m['failures'], m['scoreRate'])
    for a in c['agents']:
        m = a['ppo']
        rows[a['agent']].append((gen, m['wins'], m['losses'], m['failures'], m['scoreRate'], a['archetype']))

print('REPEATED_AGENT_WEAKNESSES')
ranked = []
for name, vals in rows.items():
    completed = sum(w + l for _, w, l, _, _, _ in vals)
    wins = sum(w for _, w, _, _, _, _ in vals)
    low_rounds = sum(rate < 0.5 for _, _, _, _, rate, _ in vals if rate is not None)
    ranked.append((wins / completed if completed else -1, low_rounds, name, vals))
for rate, low_rounds, name, vals in sorted(ranked)[:20]:
    print(name, f'pooled={rate:.3f}', f'lowRounds={low_rounds}', vals)
