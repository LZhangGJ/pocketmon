import glob
import json
from pathlib import Path

league_root = Path('/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811')
root = league_root / 'learners'
league = json.loads((league_root / 'state' / 'league.json').read_text(encoding='utf-8'))
chains = ['lucario_gold_exact', 'a02_grim_large_g9', 'a02_grim_large_g9_pokegear', 'dragapult_munkidori_large_g9', 'a08_maxbelt_large_g9']
for chain in chains:
    generation = int(league['chains'][chain]['current']['generation'])
    path = root / chain / f'generation-{generation:06d}' / 'metrics.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    print(json.dumps({
        'chain': chain,
        'generation': generation,
        'trainingConfig': data.get('trainingConfig'),
        'rolloutPlayerCounts': data.get('rolloutSummary', {}).get('playerCounts'),
        'rows': data.get('rows'),
        'decisions': data.get('decisions'),
        'epochSeat1Weights': [e.get('seat1Weight') for e in data.get('epochs', [])],
    }, sort_keys=True))
