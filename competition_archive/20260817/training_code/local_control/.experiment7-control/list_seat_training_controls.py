import json
from pathlib import Path

p = json.loads(Path('.experiment7-control/league_current_20260814.json').read_text(encoding='utf-8'))
for name, chain in p['chains'].items():
    tc = chain.get('trainingControl', {})
    rollout = tc.get('rollout', {})
    learner = tc.get('learner', {})
    print('\t'.join(map(str, [
        name,
        chain.get('current', {}).get('generation'),
        bool(tc.get('paused')),
        rollout.get('learnerSeat1Fraction', 0.5),
        learner.get('seat1Weight', 1.0),
        learner.get('normalizeAdvantagesByPlayer', False),
        learner.get('balancePlayerMinibatches', False),
    ])))
