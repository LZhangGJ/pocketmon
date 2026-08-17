from __future__ import annotations

import collections
import json
from pathlib import Path


ROOT = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(path: Path) -> dict:
    payload = read(path)
    agents = payload.get("agents", [])
    by_status = collections.Counter(row.get("pool_status", "unknown") for row in agents)
    by_archetype = collections.Counter(
        row.get("canonical_archetype") or row.get("archetype") or "unknown"
        for row in agents
    )
    identities = collections.defaultdict(list)
    for row in agents:
        deck = row.get("deck_canonical_sha256")
        policy = (
            row.get("behavior_checkpoint_sha256")
            or row.get("directory_sha256")
            or row.get("source_checkpoint")
            or row.get("name")
        )
        if deck:
            identities[(deck, policy)].append(row.get("name"))
    duplicates = [names for names in identities.values() if len(names) > 1]
    return {
        "path": str(path),
        "agents": len(agents),
        "byStatus": dict(sorted(by_status.items())),
        "byArchetype": dict(sorted(by_archetype.items())),
        "dynamicAgents": payload.get("asyncLeague", {}).get("dynamicAgents", []),
        "exactIdentityDuplicates": duplicates,
        "names": [row.get("name") for row in agents],
    }


def main() -> None:
    league = read(ROOT / "state/league.json")
    paths = [Path(league["basePool"]["path"]), Path(league["poolPath"])]
    print(json.dumps([summarize(path) for path in paths], indent=2))


if __name__ == "__main__":
    main()
