import json
from pathlib import Path

files = [
    Path('.experiment7-control/dist_latest_20260814.json'),
    Path('.experiment7-control/dist_latest_20260814_113809.json'),
    Path('.experiment7-control/dist_latest_20260814_123520.json'),
    Path('.experiment7-control/dist_latest_20260814_133525.json'),
]
targets = [
    'public_archaludon_meta',
    'champion_a02_submission4_grimmsnarl_froslass_munkidori_g000073',
    'champion_a08_dipplin_seaking_g000077',
    'champion_mega_lucario_ex_g000044',
    'persistent_a08_maxbelt_g000020',
    'persistent_a08_original_frozen_g000277',
]
for path in files:
    p = json.loads(path.read_text(encoding='utf-8'))
    c = p['chains']['dragapult_munkidori_large_g9']
    agents = {a['agent']: a['ppo'] for a in c['agents']}
    print(c['generation'], c['frozenAggregate'], c['seatMetrics'], c['ppoMinusBc'])
    print({name: agents.get(name) for name in targets})
