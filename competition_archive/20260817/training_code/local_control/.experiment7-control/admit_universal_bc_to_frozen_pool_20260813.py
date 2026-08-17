#!/usr/bin/env python3
"""Admit a fully screened Universal BC package set, preserving older BC versions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-root", type=Path, required=True)
    parser.add_argument("--integration", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--window-end", required=True)
    parser.add_argument("--screening-receipt", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.integration.resolve()))
    from async_ppo_control import atomic_write_json, build_pool_payload, read_json, state_lock, utc_now  # noqa: PLC0415

    root = args.league_root.resolve()
    league_path = root / "state/league.json"
    packages = read_json(args.package_manifest.resolve()).get("packages", [])
    screening = read_json(args.screening_receipt.resolve())
    if screening.get("status") not in {"passed", "complete_passed"}:
        raise ValueError("screening receipt has not passed")
    if not packages:
        raise ValueError("candidate has no packaged agents")

    with state_lock(league_path.with_suffix(league_path.suffix + ".lock")):
        league = read_json(league_path)
        base_path = Path(league["basePool"]["path"]).resolve()
        base = read_json(base_path)
        tag = args.window_end.replace("-", "")
        new_agents = []
        for package in packages:
            agent_dir = Path(package["agentDir"]).resolve()
            for required in (agent_dir / "main.py", agent_dir / "deck.csv"):
                if not required.is_file():
                    raise FileNotFoundError(required)
            new_agents.append(
                {
                    "name": f"universal_bc_{tag}_{args.profile}__{package['name']}",
                    "agent_dir": str(agent_dir),
                    "status": "accepted",
                    "pool_status": "frozen_universal_bc_version",
                    "archetype": package.get("archetypeId", "unknown"),
                    "canonical_archetype": str(package.get("archetypeId", "unknown")).upper(),
                    "archetype_label": package.get("archetypeLabel", "Universal BC"),
                    "skill_tier": "screened_universal_bc",
                    "policy_weight_within_archetype": 1.0,
                    "bc_window_end": args.window_end,
                    "bc_profile": args.profile,
                    "hash_verification_skipped": True,
                }
            )
        # Keep older BCs as rollback opponents; replace only an accidental duplicate of this exact version.
        names = {row["name"] for row in new_agents}
        agents = [row for row in base["agents"] if row.get("name") not in names]
        augmented = {key: value for key, value in base.items() if key != "agents"}
        augmented["agents"] = [*agents, *new_agents]
        augmented["universalBcAdmission"] = {
            "admittedAt": utc_now(),
            "windowEnd": args.window_end,
            "profile": args.profile,
            "sourceBasePool": str(base_path),
            "agents": sorted(names),
            "screeningReceipt": str(args.screening_receipt.resolve()),
        }
        output = root / "state" / f"opponent-pool-base-plus-bc-{tag}-{args.profile}.json"
        atomic_write_json(output, augmented)
        league["basePool"]["path"] = str(output)
        league["latestFrozenBcAdmission"] = augmented["universalBcAdmission"]
        atomic_write_json(Path(league["poolPath"]), build_pool_payload(league))
        atomic_write_json(league_path, league)

    daily = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc")
    receipt = daily / "admissions" / f"{args.window_end}.json"
    payload = {
        "schemaVersion": 1,
        "status": "admitted",
        **augmented["universalBcAdmission"],
        "basePool": str(output),
        "hashVerificationSkipped": True,
    }
    atomic_write_json(receipt, payload)
    print(json.dumps({"status": "admitted", "receipt": str(receipt), "agents": sorted(names)}))


if __name__ == "__main__":
    main()
