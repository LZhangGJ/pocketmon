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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--base-pool", type=Path)
    parser.add_argument("--chain", default="universal_ppo_large_256x6")
    parser.add_argument("--pool-status", default="large_g9_frozen_133_deck")
    args = parser.parse_args()

    league_path = args.league.resolve()
    with state_lock(league_path.with_suffix(league_path.suffix + ".lock")):
        league = read_json(league_path)
        base_path = (
            args.base_pool.resolve()
            if args.base_pool is not None
            else Path(league["basePool"]["path"]).resolve()
        )
        base = read_json(base_path)
        old_base_sha = sha256_file(base_path)
        changed = 0
        for row in base.get("agents", []):
            if str(row.get("pool_status")) != args.pool_status:
                continue
            if row.get("replacement_chain") != args.chain:
                row["replacement_chain"] = args.chain
                changed += 1
        matching = [
            row for row in base.get("agents", []) if row.get("pool_status") == args.pool_status
        ]
        if len(matching) != 133:
            raise ValueError(f"expected 133 Universal fallback rows, got {len(matching)}")
        if any(row.get("replacement_chain") != args.chain for row in matching):
            raise ValueError("failed to tag every Universal fallback row")
        base["updatedAt"] = utc_now()
        base["replacementPolicy"] = {
            "chain": args.chain,
            "mode": "hide all 133 fallback rows while one complete cohort is deployed",
        }
        atomic_write_json(base_path, base)
        pool_path = Path(league["poolPath"])
        atomic_write_json(pool_path, build_pool_payload(league))
        league["poolSha256"] = sha256_file(pool_path)
        league["updatedAt"] = utc_now()
        atomic_write_json(league_path, league)
        receipt = {
            "schemaVersion": 1,
            "createdAt": utc_now(),
            "chain": args.chain,
            "basePool": str(base_path),
            "oldBaseSha256": old_base_sha,
            "baseSha256": sha256_file(base_path),
            "taggedRows": len(matching),
            "changedRows": changed,
            "livePool": str(pool_path.resolve()),
            "livePoolSha256": league["poolSha256"],
            "publication": "locked atomic pool rebuild; rollout never disabled",
        }
        atomic_write_json(args.receipt.resolve(), receipt)
        print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
