from __future__ import annotations

import argparse
import json
from pathlib import Path

from async_ppo_control import (
    atomic_write_json,
    build_pool_payload,
    read_json,
    sha256_file,
    state_lock,
    utc_now,
)
from universal_deck_cohort import materialize_cohort


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atomically narrow the current Universal PPO generation to one tiered deck cohort"
    )
    parser.add_argument("--league", type=Path, required=True)
    parser.add_argument("--chain", default="universal_ppo_large_256x6")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--size", type=int, default=20)
    args = parser.parse_args()

    league_path = args.league.resolve()
    with state_lock(league_path.with_suffix(league_path.suffix + ".lock")):
        league = read_json(league_path)
        chain = league["chains"][args.chain]
        current = chain["current"]
        source_pool = Path(chain["learnerDeckPool"]).resolve()
        snapshot_id = str(current["snapshotId"])
        output_root = args.output_root.resolve() / snapshot_id
        cohort_path = output_root / "deck-cohort.json"
        cohort = materialize_cohort(
            source_pool,
            cohort_path,
            size=args.size,
            chain_name=args.chain,
            generation=int(current["generation"]),
            snapshot_sha=str(current["sha256"]),
        )

        source_manifest_path = Path(current["packageManifest"]).resolve()
        source_manifest = read_json(source_manifest_path)
        if source_manifest.get("sourcePackageManifest"):
            source_manifest_path = Path(source_manifest["sourcePackageManifest"]).resolve()
            source_manifest = read_json(source_manifest_path)
        by_deck = {
            str(package["deckSha256"]): package
            for package in source_manifest.get("packages", [])
        }
        selected_sha = [str(row["deckSha256"]) for row in cohort["selected"]]
        missing = [deck_sha for deck_sha in selected_sha if deck_sha not in by_deck]
        if missing:
            raise ValueError(f"current generation lacks selected packages: {missing}")
        cohort_manifest_path = output_root / "packages.json"
        cohort_manifest = {
            **{key: value for key, value in source_manifest.items() if key != "packages"},
            "kind": "universal-ppo-generation-cohort-packages",
            "createdAt": utc_now(),
            "sourcePackageManifest": str(source_manifest_path),
            "sourcePackageManifestSha256": sha256_file(source_manifest_path),
            "deckCohortReceipt": str(cohort_path),
            "deckCohort": {key: value for key, value in cohort.items() if key != "selected"},
            "packages": [by_deck[deck_sha] for deck_sha in selected_sha],
        }
        atomic_write_json(cohort_manifest_path, cohort_manifest)

        previous_manifest = str(current["packageManifest"])
        current["packageManifest"] = str(cohort_manifest_path)
        current["deckCohortReceipt"] = str(cohort_path)
        chain["learnerDeckCohortSize"] = args.size
        league["updatedAt"] = utc_now()
        pool_path = Path(league["poolPath"])
        atomic_write_json(pool_path, build_pool_payload(league))
        league["poolSha256"] = sha256_file(pool_path)
        atomic_write_json(league_path, league)

        receipt = {
            "schemaVersion": 1,
            "createdAt": utc_now(),
            "chain": args.chain,
            "snapshotId": snapshot_id,
            "previousPackageManifest": previous_manifest,
            "packageManifest": str(cohort_manifest_path),
            "deckCohortReceipt": str(cohort_path),
            "size": args.size,
            "tierProbabilities": cohort["tierProbabilities"],
            "tierCounts": cohort["tierCounts"],
            "deckSha256": selected_sha,
            "poolPath": str(pool_path.resolve()),
            "poolSha256": league["poolSha256"],
            "rolloutContinuity": "old pool remained active until this locked atomic publication",
        }
        atomic_write_json(output_root / "APPLIED.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
