from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from async_ppo_control import (
    atomic_write_json,
    build_pool_payload,
    read_json,
    sha256_file,
    state_lock,
    utc_now,
)


LARGE_G9_SHA = "dc532288f1916322f49596b010a83b1a02e9292cd57b978e11b8777369999b4a"
ACTIVE_CHAINS = {
    "a02_grim_large_g9_pokegear",
    "a08_maxbelt_large_g9",
    "dragapult_munkidori_large_g9",
    "lucario_gold_exact",
    "alakazam_large_g9",
    "kangaskhan_crustle_large_g9",
    "festival_grass_large_g9",
    "universal_ppo_large_256x6",
}
NEW_SPECIALISTS = {
    "alakazam_large_g9": "ALAKAZAM",
    "kangaskhan_crustle_large_g9": "KANGASKHAN_CRUSTLE",
    "festival_grass_large_g9": "FESTIVAL_LEAD_DIPPLIN",
}


def selected_by_archetype(catalog: dict[str, Any], archetype: str) -> dict[str, Any]:
    rows = [row for row in catalog["selected"] if row["archetypeId"] == archetype]
    if not rows:
        raise ValueError(f"no deck for {archetype}")
    return max(rows, key=lambda row: float(row.get("samplingWeight", 0.0)))


def package_row(package: dict[str, Any], *, pool_status: str, skill_tier: str) -> dict[str, Any]:
    agent_dir = Path(package["agentDir"])
    for required in (agent_dir / "main.py", agent_dir / "deck.csv"):
        if not required.is_file():
            raise FileNotFoundError(required)
    return {
        "name": str(package["name"]),
        "agent_dir": str(agent_dir.resolve()),
        "status": "accepted",
        "pool_status": pool_status,
        "archetype": str(package["archetypeId"]),
        "canonical_archetype": str(package["archetypeId"]).upper(),
        "archetype_label": str(package.get("archetypeLabel") or package["archetypeId"]),
        "deck_canonical_sha256": str(package["deckSha256"]),
        "directory_sha256": str(package["directorySha256"]),
        "skill_tier": skill_tier,
        "policy_weight_within_archetype": 1.0,
    }


def specialist_chain(name: str, deck: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    return {
        "deckName": str(deck["name"]),
        "archetypeId": str(deck["archetypeId"]),
        "archetypeLabel": f"{deck['archetypeLabel']} / Universal Large g9 specialist",
        "deckPath": str(Path(deck["deckPath"]).resolve()),
        "deckSha256": str(deck["deckSha256"]),
        "teacher": str(checkpoint.resolve()),
        "current": {
            "generation": 0,
            "checkpoint": str(checkpoint.resolve()),
            "sha256": LARGE_G9_SHA,
            "snapshotId": f"{name}-g000000-{LARGE_G9_SHA[:12]}",
        },
        "history": [],
        "poolControl": {"enabled": True, "mode": "atomic_generation_replace"},
        "trainingControl": {
            "schemaVersion": 2,
            "updatedAt": utc_now(),
            "sourceRoundId": "large-g9-pool-reconfiguration-20260814",
            "evidence": {
                "engineSeedControlled": False,
                "reason": "user-directed independent Large g9 specialist",
            },
            "rollout": {
                "enabled": True,
                "selfPlayFraction": 0.15,
                "learnerSeat1Fraction": 0.55,
                "archetypeWeights": {
                    "A02": 1.35,
                    "A08": 1.25,
                    "ALAKAZAM": 1.35,
                    "DRAGAPULT": 1.35,
                    "KANGASKHAN_CRUSTLE": 1.35,
                    "FESTIVAL_LEAD_DIPPLIN": 1.35,
                    "LUCARIO_GOLD": 1.2,
                },
                "agentWeights": {},
                "weightPolicy": "specialist coverage; bounded until complete arena evidence",
            },
            "learner": {
                "minDecisions": 12000,
                "maxBehaviorLag": 2,
                "teacherAnchorCoefficient": 0.05,
                "seat1Weight": 1.25,
                "learningRate": 5e-6,
                "ppoEpochs": 1,
                "normalizeAdvantagesByPlayer": True,
                "balancePlayerMinibatches": True,
            },
        },
        "bootstrap": {
            "reason": "user-directed Universal Large g9 initialization",
            "sourceSnapshotId": "universal_ppo_large_256x6-g000009-dc532288f191",
            "sourceCheckpointSha256": LARGE_G9_SHA,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--large-packages", type=Path, required=True)
    parser.add_argument("--large-checkpoint", type=Path, required=True)
    parser.add_argument("--best-a02-packages", type=Path, required=True)
    parser.add_argument("--best-a08-packages", type=Path, required=True)
    parser.add_argument("--base-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    if sha256_file(args.large_checkpoint) != LARGE_G9_SHA:
        raise ValueError("Universal Large g9 checkpoint SHA mismatch")
    catalog = read_json(args.catalog)
    selected = catalog.get("selected", [])
    if len(selected) != 133 or any(not Path(row["deckPath"]).is_file() for row in selected):
        raise ValueError("expected 133 complete catalog decks")
    catalog_by_sha = {str(row["deckSha256"]): row for row in selected}
    large_manifest = read_json(args.large_packages)
    packages = large_manifest.get("packages", [])
    if len(packages) != 133:
        raise ValueError(f"expected 133 Large g9 packages, got {len(packages)}")
    package_shas = {str(row["deckSha256"]) for row in packages}
    if package_shas != set(catalog_by_sha):
        raise ValueError("Large g9 package/catalog deck mismatch")

    base_agents = []
    for package in packages:
        row = package_row(
            package,
            pool_status="large_g9_frozen_133_deck",
            skill_tier="universal_large_g9",
        )
        evidence = catalog_by_sha[row["deck_canonical_sha256"]]
        row["evidence_tier"] = evidence.get("evidenceTier")
        row["source_sampling_weight"] = evidence.get("samplingWeight")
        row["replacement_chain"] = "universal_ppo_large_256x6"
        base_agents.append(row)

    retained = []
    for label, manifest_path in (
        ("best_a02", args.best_a02_packages),
        ("best_a08", args.best_a08_packages),
    ):
        manifest = read_json(manifest_path)
        rows = manifest.get("packages", [])
        if len(rows) != 1:
            raise ValueError(f"expected one {label} package")
        row = package_row(rows[0], pool_status="retained_best_specialist", skill_tier=label)
        row["name"] = f"retained_{label}__{row['name']}"
        base_agents.append(row)
        retained.append(row["name"])

    if any(token in row["name"].lower() for row in base_agents for token in ("notebook", "public_")):
        raise ValueError("public/notebook agent leaked into rebuilt base")
    base_payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "policy": "133 Universal Large g9 deck agents plus retained best A02/A08 anchors",
        "sourceLargePackages": {
            "path": str(args.large_packages.resolve()),
            "sha256": sha256_file(args.large_packages),
        },
        "agents": base_agents,
    }
    atomic_write_json(args.base_output, base_payload)

    lock = args.league.with_suffix(args.league.suffix + ".lock")
    with state_lock(lock):
        league = read_json(args.league)
        backup_root = args.receipt.parent / "pre-large-g9-pool-backup"
        backup_root.mkdir(parents=True, exist_ok=True)
        for source, label in (
            (args.league, "league.json"),
            (Path(league["poolPath"]), "opponent-pool-live.json"),
            (Path(league["basePool"]["path"]), "opponent-pool-base.json"),
        ):
            target = backup_root / label
            if not target.exists():
                shutil.copy2(source, target)
        checkpoint = args.large_checkpoint.resolve()
        for name, chain in league["chains"].items():
            chain["poolControl"] = {
                "enabled": name in ACTIVE_CHAINS,
                "mode": "atomic_generation_replace",
            }
            rollout = chain.setdefault("trainingControl", {}).setdefault("rollout", {})
            if name not in ACTIVE_CHAINS:
                rollout["enabled"] = False

        for name, archetype in NEW_SPECIALISTS.items():
            deck = selected_by_archetype(catalog, archetype)
            if name not in league["chains"]:
                league["chains"][name] = specialist_chain(name, deck, checkpoint)
            else:
                existing = league["chains"][name]
                if str(existing["deckSha256"]) != str(deck["deckSha256"]):
                    raise ValueError(f"existing {name} deck mismatch")
                existing["poolControl"] = {"enabled": True, "mode": "atomic_generation_replace"}
                existing.setdefault("trainingControl", {}).setdefault("rollout", {})["enabled"] = True

        for name in ACTIVE_CHAINS:
            if name not in league["chains"]:
                raise ValueError(f"missing active chain {name}")
            chain = league["chains"][name]
            chain["poolControl"] = {"enabled": True, "mode": "atomic_generation_replace"}
            chain.setdefault("trainingControl", {}).setdefault("rollout", {})["enabled"] = True
            chain.get("trainingControl", {}).pop("paused", None)

        universal = league["chains"]["universal_ppo_large_256x6"]
        universal["learnerDeckPool"] = str(args.catalog.resolve())
        universal["learnerDeckCohortSize"] = 20
        # The immutable g9 policy is already materialized for all 133 decks.
        # Point its current snapshot at that complete manifest immediately;
        # later generations will publish their own 133-package manifests.
        universal["current"]["packageManifest"] = str(args.large_packages.resolve())
        league["basePool"] = {"path": str(args.base_output.resolve())}
        league["updatedAt"] = utc_now()
        pool_path = Path(league["poolPath"])
        pool_payload = build_pool_payload(league)
        atomic_write_json(pool_path, pool_payload)
        league["poolSha256"] = sha256_file(pool_path)
        league["largeG9PoolReconfiguration"] = {
            "at": league["updatedAt"],
            "baseAgents": len(base_agents),
            "activeChains": sorted(ACTIVE_CHAINS),
            "policy": "old pool remains readable until atomic replacement",
        }
        atomic_write_json(args.league, league)

    receipt = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "league": str(args.league.resolve()),
        "basePool": str(args.base_output.resolve()),
        "baseAgents": len(base_agents),
        "largeDeckAgents": len(packages),
        "retained": retained,
        "removedCategories": ["public notebook", "Hard Exploiter", "Diversity", "old champions", "old BC deck agents"],
        "activeChains": sorted(ACTIVE_CHAINS),
        "atomicReplacement": True,
        "livePoolSha256": sha256_file(Path(read_json(args.league)["poolPath"])),
    }
    atomic_write_json(args.receipt, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
