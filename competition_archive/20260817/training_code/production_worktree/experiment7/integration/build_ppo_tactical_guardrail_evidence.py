from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate exact-snapshot tactical opportunity/error evidence"
    )
    parser.add_argument("--buffer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-revision", type=int, default=11)
    args = parser.parse_args()

    ready = args.buffer_root.resolve() / "ready"
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    seen_outputs: set[str] = set()
    for summary_path in sorted(ready.glob("*/*.jsonl.gz.summary.json")):
        try:
            summary = read_json(summary_path)
        except (OSError, ValueError, TypeError):
            continue
        control = summary.get("samplingControl") or {}
        revision = int(control.get("tacticalShapingRevision", 0) or 0)
        if revision < args.minimum_revision:
            continue
        snapshot_id = str(summary.get("behaviorSnapshotId", ""))
        chain = summary_path.parent.name
        output = summary.get("output") or {}
        dedupe_key = str(output.get("sha256") or output.get("path") or summary_path)
        if not snapshot_id or dedupe_key in seen_outputs:
            continue
        seen_outputs.add(dedupe_key)
        group = groups.setdefault(
            (chain, snapshot_id),
            {
                "chain": chain,
                "snapshotId": snapshot_id,
                "minimumRevision": revision,
                "maximumRevision": revision,
                "shards": 0,
                "episodes": 0,
                "decisions": 0,
                "opportunities": Counter(),
                "errors": Counter(),
                "receipts": [],
            },
        )
        group["minimumRevision"] = min(group["minimumRevision"], revision)
        group["maximumRevision"] = max(group["maximumRevision"], revision)
        group["shards"] += 1
        group["episodes"] += int(summary.get("episodes", 0) or 0)
        group["decisions"] += int(summary.get("decisions", 0) or 0)
        for opportunity, row in (summary.get("tacticalOpportunityRates") or {}).items():
            group["opportunities"][opportunity] += int(row.get("opportunities", 0) or 0)
            group["errors"][opportunity] += int(row.get("errors", 0) or 0)
        group["receipts"].append(str(summary_path.resolve()))

    snapshots: dict[str, dict[str, Any]] = {}
    for (chain, snapshot_id), group in sorted(groups.items()):
        metrics = {}
        total_opportunities = 0
        total_errors = 0
        for opportunity in sorted(group["opportunities"]):
            count = int(group["opportunities"][opportunity])
            errors = int(group["errors"][opportunity])
            total_opportunities += count
            total_errors += errors
            metrics[opportunity] = {
                "opportunities": count,
                "errors": errors,
                "errorRate": errors / count if count else None,
            }
        snapshots.setdefault(chain, {})[snapshot_id] = {
            "chain": chain,
            "snapshotId": snapshot_id,
            "minimumRevision": group["minimumRevision"],
            "maximumRevision": group["maximumRevision"],
            "shards": group["shards"],
            "episodes": group["episodes"],
            "decisions": group["decisions"],
            "trackedOpportunities": total_opportunities,
            "errors": total_errors,
            "aggregateErrorRate": (
                total_errors / total_opportunities if total_opportunities else None
            ),
            "opportunityRates": metrics,
            "receipts": group["receipts"],
        }
    payload = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "bufferRoot": str(ready.parent),
        "minimumRevision": args.minimum_revision,
        "snapshotCount": sum(len(rows) for rows in snapshots.values()),
        "snapshots": snapshots,
    }
    write_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
