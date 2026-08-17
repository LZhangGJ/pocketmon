import gzip
import json
from collections import Counter
from pathlib import Path

league_root = Path('/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811')
league = json.loads((league_root / 'state' / 'league.json').read_text(encoding='utf-8'))
chains = ['lucario_gold_exact', 'a02_grim_large_g9', 'a02_grim_large_g9_pokegear', 'dragapult_munkidori_large_g9', 'a08_maxbelt_large_g9']
for chain in chains:
    generation = int(league['chains'][chain]['current']['generation'])
    metrics = json.loads((league_root / 'learners' / chain / f'generation-{generation:06d}' / 'metrics.json').read_text(encoding='utf-8'))
    counts = Counter()
    for rollout in metrics.get('rollouts', []):
        with gzip.open(rollout['path'], 'rt', encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                if 'player' in row:
                    counts[int(row['player'])] += 1
    total = sum(counts.values())
    print(json.dumps({'chain': chain, 'generation': generation, 'playerDecisionCounts': counts, 'seat1DecisionFraction': counts[1] / total if total else None, 'trainingConfig': metrics.get('trainingConfig')}, sort_keys=True))
