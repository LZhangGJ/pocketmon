import collections
import json
from pathlib import Path


cards = {
    int(card["cardId"]): card
    for card in json.loads(Path("data/reference/official_cards.json").read_text(encoding="utf-8"))
}
counts = collections.Counter(
    map(int, Path(".experiment7-control/universal_large_g9_deck.csv").read_text().split())
)
for card_id, count in sorted(counts.items()):
    card = cards[card_id]
    print(f"{count}x\t{card_id}\t{card['name']}\ttype={card['cardType']}")
