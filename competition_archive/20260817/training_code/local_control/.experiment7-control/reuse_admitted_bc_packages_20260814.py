from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")
INTEGRATION = Path("/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0/experiment7/integration")
sys.path.insert(0, str(INTEGRATION))
from async_ppo_control import atomic_write_json, read_json, state_lock, utc_now  # noqa: E402


def main() -> None:
    league_path = ROOT / "state/league.json"
    league = read_json(league_path)
    base = read_json(Path(league["basePool"]["path"]))
    by_name = {row["name"]: row for row in base["agents"]}
    specs = {
        "universal_ppo_standard_1m": (
            "universal_bc_20260812_standard_1m__universal_bc_baseline__a02_g247_baseline",
            "universal_standard_a02_representative",
            "UNIVERSAL_STANDARD",
            "Universal standard 1.05M / 133-deck evolving PPO",
        ),
        "universal_ppo_large_256x6": (
            "universal_bc_20260812_large_256x6__universal_bc_baseline__a08_rabsca",
            "universal_large_a08_representative",
            "UNIVERSAL_LARGE",
            "Universal large 6.44M / 133-deck evolving PPO",
        ),
    }
    receipts = []
    for chain_name, (agent_name, deck_name, archetype_id, archetype_label) in specs.items():
        row = by_name[agent_name]
        agent_dir = Path(row["agent_dir"])
        deck_sha = row["deck_canonical_sha256"]
        # Use the immutable admitted package itself for g0. Future generations
        # are exported normally from PPO checkpoints.
        manifest = {
            "schemaVersion": 1,
            "createdAt": utc_now(),
            "packages": [{
                "name": f"live_{chain_name}_g000000__{deck_name}",
                "agentDir": str(agent_dir),
                "deckSha256": deck_sha,
                "archetypeId": archetype_id,
                "archetypeLabel": archetype_label,
                "directorySha256": row["directory_sha256"],
            }],
        }
        target = ROOT / f"learners/{chain_name}/generation-000000-bootstrap/deployment/packages/packages.json"
        atomic_write_json(target, manifest)
        # Find the exact immutable 60-card deck from the pool manifest.
        deck_pool = read_json(ROOT / "control/universal-bc-deck-pool-20260813/universal_bc_decks.json")
        deck = next(item for item in deck_pool["selected"] if item["deckSha256"] == deck_sha)
        receipts.append({"chain": chain_name, "reusedAgent": agent_name, "manifest": str(target)})
    with state_lock(league_path.with_suffix(league_path.suffix + ".lock")):
        latest = read_json(league_path)
        for chain_name, (agent_name, deck_name, archetype_id, archetype_label) in specs.items():
            row = by_name[agent_name]
            deck_sha = row["deck_canonical_sha256"]
            deck_pool = read_json(ROOT / "control/universal-bc-deck-pool-20260813/universal_bc_decks.json")
            deck = next(item for item in deck_pool["selected"] if item["deckSha256"] == deck_sha)
            latest["chains"][chain_name].update({
                "deckName": deck_name,
                "deckPath": deck["deckPath"],
                "deckSha256": deck_sha,
                "archetypeId": archetype_id,
                "archetypeLabel": archetype_label,
            })
        atomic_write_json(league_path, latest)
    receipt = {"schemaVersion": 1, "createdAt": utc_now(), "status": "reused_admitted_packages", "rows": receipts}
    atomic_write_json(ROOT / "control/seven-ppo-distributed-20260814/reuse-admitted-package-receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
