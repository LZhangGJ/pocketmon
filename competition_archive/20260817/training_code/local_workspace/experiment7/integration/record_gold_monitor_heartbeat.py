#!/usr/bin/env python3
"""Atomically attach the latest compact evidence to the gold monitor ledgers."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def read_object(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--latest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--event", required=True)
    args = parser.parse_args()

    compact = read_object(args.compact)
    observed_at = datetime.now(timezone.utc).isoformat()
    evidence = {
        "observedAt": observed_at,
        "event": args.event,
        "compactSnapshot": str(args.compact.resolve()),
        "leagueUpdatedAt": compact.get("leagueUpdatedAt"),
        "poolSha256": compact.get("poolSha256"),
        "generations": {
            chain: state.get("generation")
            for chain, state in compact.get("chains", {}).items()
        },
    }
    for path in (args.latest, args.plan):
        payload = read_object(path)
        payload["lastHeartbeat"] = evidence
        write_atomic(path, payload)


if __name__ == "__main__":
    main()
