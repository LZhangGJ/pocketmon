from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Install a validated PPO branch as a persistent league opponent")
    parser.add_argument("--league-root", type=Path, required=True)
    parser.add_argument("--integration", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--package-name", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archetype-id", default="A08")
    parser.add_argument("--archetype-label", default="A08 Maximum Belt")
    parser.add_argument("--screening-json", type=Path)
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
    registry_path = root / "state/persistent-ppo-branches.json"
    augmented_base = root / "state/opponent-pool-base-plus-persistent-branches.json"
    agent_name = f"persistent_{args.branch}_g{args.generation:06d}"

    manifest = read_json(args.package_manifest.resolve())
    packages = [row for row in manifest.get("packages", []) if row.get("name") == args.package_name]
    if len(packages) != 1:
        raise RuntimeError(f"expected one package named {args.package_name}: {packages}")
    package = packages[0]
    agent_dir = Path(package["agentDir"]).resolve()
    checkpoint = args.checkpoint.resolve()
    for required in (agent_dir / "main.py", agent_dir / "deck.csv", checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)

    screening = read_json(args.screening_json.resolve()) if args.screening_json else {}
    persistent = {
        "name": agent_name,
        "agent_dir": str(agent_dir),
        "status": "accepted",
        "pool_status": "persistent_ppo_branch",
        "archetype": args.archetype_id,
        "canonical_archetype": args.archetype_id.upper(),
        "archetype_label": args.archetype_label,
        "skill_tier": "persistent_ppo_branch",
        "policy_weight_within_archetype": 1.0,
        "ppo_branch": args.branch,
        "ppo_generation": args.generation,
        "source_checkpoint": str(checkpoint),
        "screening": screening,
    }

    lock_path = league_path.with_suffix(league_path.suffix + ".lock")
    with state_lock(lock_path):
        league = read_json(league_path)
        current_base = Path(league["basePool"]["path"]).resolve()
        base = read_json(current_base)
        agents = [
            row
            for row in base["agents"]
            if not (
                row.get("pool_status") == "persistent_ppo_branch"
                and row.get("ppo_branch") == args.branch
            )
        ]
        augmented = {key: value for key, value in base.items() if key != "agents"}
        augmented["agents"] = [*agents, persistent]
        augmented["persistentPpoBranches"] = {
            "createdAt": utc_now(),
            "sourceBasePool": str(current_base),
            "sampling": "B08 uniform canonical archetype, then uniform agent within archetype",
            "agents": [
                row["name"]
                for row in augmented["agents"]
                if row.get("pool_status") == "persistent_ppo_branch"
            ],
        }
        atomic_write_json(augmented_base, augmented)

        previous_registry = read_json(registry_path) if registry_path.is_file() else {}
        records = [
            row
            for row in previous_registry.get("branches", [])
            if row.get("branch") != args.branch
        ]
        record = {
            "branch": args.branch,
            "agent": agent_name,
            "generation": args.generation,
            "packageManifest": str(args.package_manifest.resolve()),
            "agentDir": str(agent_dir),
            "checkpoint": str(checkpoint),
            "installedAt": utc_now(),
            "replacementPolicy": "replace only after a newer checkpoint passes the full branch gate",
        }
        records.append(record)
        registry = {
            "schemaVersion": 1,
            "updatedAt": utc_now(),
            "augmentedBasePool": str(augmented_base),
            "hashVerificationSkipped": True,
            "branches": records,
        }
        atomic_write_json(registry_path, registry)
        league["basePool"]["path"] = str(augmented_base)
        league["persistentPpoBranchRegistryPath"] = str(registry_path)
        atomic_write_json(league_path, league)
        atomic_write_json(Path(league["poolPath"]), build_pool_payload(league))

    receipt = root / "monitoring/persistent-ppo-branches" / f"{args.branch}-g{args.generation:06d}.json"
    atomic_write_json(receipt, {"schemaVersion": 1, **record, "poolPath": str(league["poolPath"])})
    print(json.dumps({"status": "installed", "record": record, "receipt": str(receipt)}))


if __name__ == "__main__":
    main()
