from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")
league = json.loads((ROOT / "state/league.json").read_text())
print("CHAINS", list(league.get("chains", {})))
for name, chain in league.get("chains", {}).items():
    print("CHAIN", name)
    print(json.dumps(chain, ensure_ascii=False, sort_keys=True))
print("OPPONENT_POOL", json.dumps(league.get("opponentPool", {}), ensure_ascii=False, sort_keys=True))

print("WORKERS")
for pidfile in sorted((ROOT / "workers").glob("*.pid")):
    print(pidfile.name, pidfile.read_text().strip())

training_root = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/0812-d14-ram-npz-fast-20260813")
print("MODELS")
for path in sorted(training_root.rglob("best_model.pt")):
    print(path, path.stat().st_size)
for path in sorted(training_root.rglob("*raw-best*")):
    if path.is_file():
        print(path, path.stat().st_size)

pool_root = ROOT / "control/universal-bc-deck-pool-20260813"
print("DECK_MANIFESTS")
for path in sorted(pool_root.rglob("*.json")):
    print(path)
manifest = json.loads((pool_root / "universal_bc_decks.json").read_text())
print("DECK_POOL_KEYS", list(manifest))
print("DECK_POOL_SAMPLE", json.dumps(manifest.get("selected", [])[:5], ensure_ascii=False, indent=2))
