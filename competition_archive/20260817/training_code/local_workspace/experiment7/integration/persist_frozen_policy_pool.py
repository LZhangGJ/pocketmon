from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from async_ppo_control import (
        atomic_write_json,
        build_pool_payload,
        read_json,
    )
    from pool_reconciler import register_source_and_reconcile
except ModuleNotFoundError:
    from .async_ppo_control import (
        atomic_write_json,
        build_pool_payload,
        read_json,
    )
    from .pool_reconciler import register_source_and_reconcile


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def policy_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("deck_canonical_sha256") or row.get("deckSha256") or ""),
        str(row.get("policyVersion") or row.get("behavior_checkpoint_sha256") or ""),
    )


def persist_frozen_pool(
    league_path: Path,
    pool_path: Path,
    source_kind: str,
    receipt_path: Path,
) -> dict[str, Any]:
    league_path = league_path.resolve()
    pool_path = pool_path.resolve()
    receipt_path = receipt_path.resolve()
    frozen = read_json(pool_path)
    candidates = [
            row
            for row in frozen.get("agents", [])
            if row.get("sourceKind") == source_kind
            and row.get("immutable") is True
            and row.get("ppoUpdatesAllowed") is False
        ]
    if not candidates:
        raise ValueError("frozen policy pool has no admissible agents")
    result = register_source_and_reconcile(
        league_path, pool_path, source_kind, build_live=build_pool_payload
    )
    observed = utc_now()
    receipt = {
            "schemaVersion": 1,
            "status": "persisted",
            "observedAt": observed,
            "league": str(league_path),
            "frozenPool": {
                "path": str(pool_path),
                "sourceKind": source_kind,
                "agents": len(candidates),
            },
            "basePool": result["basePool"],
            "livePool": {
                "path": result["livePool"]["path"],
                "after": result["livePool"]["agents"],
                "sha256": result["livePool"]["sha256"],
            },
            "dedupe": "deck_identity_plus_policy_version",
        }
    atomic_write_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--source-kind", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            persist_frozen_pool(args.league, args.pool, args.source_kind, args.receipt)
        )
    )


if __name__ == "__main__":
    main()
