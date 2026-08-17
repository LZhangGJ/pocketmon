import collections
import json
from pathlib import Path

catalog = json.loads(Path('.experiment7-control/engine_catalog.json').read_text(encoding='utf-8'))
cards = {int(c['cardId']): c for c in catalog['cards']}
ids = [int(x) for x in Path('.experiment7-control/dragapult_munkidori_deck.csv').read_text().split()]
for cid, count in collections.Counter(ids).items():
    c = cards[cid]
    print(count, cid, c['name'], 'type', c['cardType'], 'hp', c['hp'], 'stage1', c['stage1'], 'stage2', c['stage2'], 'ex', c['ex'], 'evolvesFrom', c['evolvesFrom'])
    for a in c.get('attacks', []):
        print('  attack', a)
    for s in c.get('skills', []):
        print('  skill', s)
