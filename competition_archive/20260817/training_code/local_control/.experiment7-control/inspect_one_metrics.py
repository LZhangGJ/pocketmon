import json
from pathlib import Path

path = Path('/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811/learners/lucario_gold_exact/generation-000065/metrics.json')
p = json.loads(path.read_text(encoding='utf-8'))
print(sorted(p))
for key in ('trainingConfig', 'input', 'inputs', 'rollouts', 'rolloutPaths', 'sourceRollouts', 'dataset', 'examples', 'epochs'):
    if key in p:
        print(key, json.dumps(p[key], ensure_ascii=False)[:5000])
