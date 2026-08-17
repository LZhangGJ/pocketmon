import json
from pathlib import Path

p = json.loads(Path('.experiment7-control/league_current_20260814.json').read_text(encoding='utf-8'))
for name, chain in p['chains'].items():
    current = chain['current']
    paused = bool(chain.get('trainingControl', {}).get('paused'))
    print(f"{name}\tg{current['generation']}\tpaused={paused}\t{current['snapshotId']}")
