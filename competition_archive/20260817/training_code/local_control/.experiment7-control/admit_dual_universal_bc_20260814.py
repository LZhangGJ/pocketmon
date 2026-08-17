from __future__ import annotations

import json
import os
import sys
from pathlib import Path


MAIN = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")
SCREENING = Path(
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/"
    "0812-d14-ram-npz-fast-20260813/replacement-screening"
)
INTEGRATION = Path("/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0/experiment7/integration")
MANIFESTS = {
    "standard_1m": SCREENING / (
        "standard_1m-frozen40/monitoring/full-matrix/universal-bc-baseline/packages/packages.json"
    ),
    "large_256x6": SCREENING / (
        "large_256x6-frozen40/monitoring/full-matrix/universal-bc-baseline/packages/packages.json"
    ),
}
WINDOW_END = "2026-08-12"


def main() -> None:
    sys.path.insert(0, str(INTEGRATION))
    from async_ppo_control import (  # noqa: PLC0415
        atomic_write_json,
        build_pool_payload,
        read_json,
        state_lock,
        utc_now,
    )

    packages_by_profile = {
        profile: read_json(path).get("packages", []) for profile, path in MANIFESTS.items()
    }
    for profile, packages in packages_by_profile.items():
        if len(packages) != 4:
            raise RuntimeError(f"{profile} expected four representative packages: {packages}")
        for package in packages:
            agent_dir = Path(package["agentDir"]).resolve()
            for required in (agent_dir / "main.py", agent_dir / "deck.csv"):
                if not required.is_file():
                    raise FileNotFoundError(required)

    league_path = MAIN / "state/league.json"
    with state_lock(league_path.with_suffix(league_path.suffix + ".lock")):
        league = read_json(league_path)
        source_base = Path(league["basePool"]["path"]).resolve()
        base = read_json(source_base)
        removed = [
            row["name"]
            for row in base["agents"]
            if row.get("pool_status") == "frozen_universal_bc_version"
            or row.get("name") == "team_submission_4_portable_bc"
        ]
        retained = [row for row in base["agents"] if row.get("name") not in set(removed)]
        admitted = []
        for profile, packages in packages_by_profile.items():
            for package in packages:
                admitted.append(
                    {
                        "name": f"universal_bc_20260812_{profile}__{package['name']}",
                        "agent_dir": str(Path(package["agentDir"]).resolve()),
                        "status": "accepted",
                        "pool_status": "frozen_universal_bc_version",
                        "archetype": package.get("archetypeId", "unknown"),
                        "canonical_archetype": str(package.get("archetypeId", "unknown")).upper(),
                        "archetype_label": package.get("archetypeLabel", "Universal BC"),
                        "deck_canonical_sha256": package.get("deckSha256", ""),
                        "directory_sha256": package.get("directorySha256", ""),
                        "skill_tier": "user_admitted_universal_bc",
                        "policy_weight_within_archetype": 1.0,
                        "bc_window_end": WINDOW_END,
                        "bc_profile": profile,
                        "admission_basis": "user-approved dual replacement after parity, smoke, frozen40 and direct evidence",
                        "hash_verification_skipped": True,
                    }
                )
        output = MAIN / "state/opponent-pool-base-plus-bc-20260812-dual.json"
        augmented = {key: value for key, value in base.items() if key != "agents"}
        augmented["agents"] = [*retained, *admitted]
        augmented["universalBcAdmission"] = {
            "admittedAt": utc_now(),
            "windowEnd": WINDOW_END,
            "profiles": list(MANIFESTS),
            "sourceBasePool": str(source_base),
            "removedOldBcAgents": removed,
            "agents": [row["name"] for row in admitted],
            "decision": "explicit_user_override_admit_both_and_replace_old_bc",
            "engineSeedControlled": False,
        }
        atomic_write_json(output, augmented)
        league["basePool"]["path"] = str(output)
        league["latestFrozenBcAdmission"] = augmented["universalBcAdmission"]
        atomic_write_json(Path(league["poolPath"]), build_pool_payload(league))
        atomic_write_json(league_path, league)

    admission_root = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/admissions")
    receipt = admission_root / "2026-08-12-dual.json"
    atomic_write_json(
        receipt,
        {
            "schemaVersion": 1,
            "status": "admitted",
            **augmented["universalBcAdmission"],
            "basePool": str(output),
            "representativeAgentCount": len(admitted),
            "frozenEngineProfiles": list(MANIFESTS),
        },
    )
    print(
        json.dumps(
            {
                "status": "admitted",
                "receipt": str(receipt),
                "removed": removed,
                "admitted": [row["name"] for row in admitted],
                "baseAgentCount": len(augmented["agents"]),
            }
        )
    )


if __name__ == "__main__":
    main()
