from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Add frozen PPO champions to the B08 opponent pool")
    parser.add_argument("--league-root", type=Path, required=True)
    parser.add_argument("--integration", type=Path, required=True)
    parser.add_argument("--champion", action="append", required=True, help="CHAIN=GENERATION")
    args = parser.parse_args()

    sys.path.insert(0, str(args.integration.resolve()))
    from async_ppo_control import (  # noqa: PLC0415
        atomic_write_json,
        build_pool_payload,
        read_json,
        state_lock,
        utc_now,
    )

    root = args.league_root.resolve()
    league_path = root / "state/league.json"
    registry_path = root / "state/champions.json"
    augmented_base = root / "state/opponent-pool-base-plus-champions.json"
    requested = {}
    for value in args.champion:
        chain, generation = value.rsplit("=", 1)
        requested[chain] = int(generation)

    lock_path = league_path.with_suffix(league_path.suffix + ".lock")
    with state_lock(lock_path):
        league = read_json(league_path)
        previous_registry = read_json(registry_path) if registry_path.is_file() else {}
        source_base = Path(
            previous_registry.get("sourceBasePool", league["basePool"]["path"])
        ).resolve()
        base = read_json(source_base)
        base_agents = [
            row for row in base["agents"] if row.get("pool_status") != "historical_ppo_champion"
        ]
        champions = []
        for chain_name, generation in sorted(requested.items()):
            chain = league["chains"][chain_name]
            generation_root = root / "learners" / chain_name / f"generation-{generation:06d}"
            manifest_path = generation_root / "deployment/packages/packages.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest = read_json(manifest_path)
            packages = [
                row
                for row in manifest["packages"]
                if str(row.get("archetypeId", "")).upper()
                == str(chain["archetypeId"]).upper()
            ]
            if len(packages) != 1:
                raise RuntimeError(f"expected one package for {chain_name} g{generation}: {packages}")
            package = packages[0]
            agent_dir = Path(package["agentDir"]).resolve()
            for required in (agent_dir / "main.py", agent_dir / "deck.csv"):
                if not required.is_file():
                    raise FileNotFoundError(required)
            champions.append(
                {
                    "name": f"champion_{chain_name}_g{generation:06d}",
                    "agent_dir": str(agent_dir),
                    "status": "accepted",
                    "pool_status": "historical_ppo_champion",
                    "archetype": str(chain["archetypeId"]),
                    "canonical_archetype": str(chain["archetypeId"]).upper(),
                    "archetype_label": str(chain["archetypeLabel"]),
                    "skill_tier": "champion_ppo",
                    "policy_weight_within_archetype": 1.0,
                    "ppo_chain": chain_name,
                    "ppo_generation": generation,
                }
            )
        augmented = {key: value for key, value in base.items() if key != "agents"}
        augmented["agents"] = [*base_agents, *champions]
        augmented["championPool"] = {
            "createdAt": utc_now(),
            "sampling": "B08 uniform canonical archetype; champion is an additional agent within its archetype",
            "champions": [row["name"] for row in champions],
        }
        atomic_write_json(augmented_base, augmented)
        registry = {
            "schemaVersion": 1,
            "createdAt": utc_now(),
            "sourceBasePool": str(source_base),
            "augmentedBasePool": str(augmented_base),
            "promotionPolicy": {
                "automatic": False,
                "minimumConsecutiveRounds": 2,
                "aggregateRegressionThresholdPp": -5.0,
                "directBcMinimumScoreRate": 0.45,
            },
            "champions": champions,
            "hashVerificationSkipped": True,
        }
        atomic_write_json(registry_path, registry)
        league["basePool"]["path"] = str(augmented_base)
        league["championRegistryPath"] = str(registry_path)
        atomic_write_json(league_path, league)
        atomic_write_json(Path(league["poolPath"]), build_pool_payload(league))
    print(json.dumps(registry))


if __name__ == "__main__":
    main()
