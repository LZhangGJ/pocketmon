from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


KEY_FIELDS = ("learner", "opponent", "seed", "learner_seat")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    try:
        return tuple(str(row[field]) for field in KEY_FIELDS)
    except KeyError as exc:
        raise ValueError(f"CSV row is missing key field {exc.args[0]!r}: {row}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an immutable Arena resume schedule by subtracting completed result keys"
    )
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-shards", type=int, nargs="+", required=True)
    parser.add_argument("--source-shard-count", type=int, required=True)
    parser.add_argument("--expect-pending", type=int)
    args = parser.parse_args()

    schedule = args.schedule.resolve()
    result_paths = [path.resolve() for path in args.results]
    output = args.output.resolve()
    receipt = args.receipt.resolve()
    if output.exists() or receipt.exists():
        raise FileExistsError(f"refusing to overwrite resume artifacts: {output} / {receipt}")
    if args.source_shard_count <= 0:
        raise ValueError("source shard count must be positive")
    selected_shards = sorted(set(args.source_shards))
    if not selected_shards or any(
        shard < 0 or shard >= args.source_shard_count for shard in selected_shards
    ):
        raise ValueError("source shards must be unique values in [0, source_shard_count)")

    fieldnames, schedule_rows = read_csv(schedule)
    schedule_keys = [row_key(row) for row in schedule_rows]
    if len(schedule_keys) != len(set(schedule_keys)):
        raise ValueError("source schedule contains duplicate Arena keys")
    selected_rows = [
        row
        for index, row in enumerate(schedule_rows)
        if index % args.source_shard_count in selected_shards
    ]
    selected_keys = {row_key(row) for row in selected_rows}

    completed_keys: set[tuple[str, ...]] = set()
    result_receipts = []
    for path in result_paths:
        _, rows = read_csv(path)
        keys = {row_key(row) for row in rows}
        unknown = keys - set(schedule_keys)
        if unknown:
            raise ValueError(f"result contains keys outside source schedule: {path}: {sorted(unknown)[:3]}")
        duplicate_completed = completed_keys & keys
        if duplicate_completed:
            raise ValueError(
                f"duplicate completed keys across result files: {path}: {sorted(duplicate_completed)[:3]}"
            )
        completed_keys.update(keys)
        result_receipts.append(
            {"path": str(path), "sha256": sha256_file(path), "rows": len(rows)}
        )

    pending_rows = [row for row in selected_rows if row_key(row) not in completed_keys]
    if args.expect_pending is not None and len(pending_rows) != args.expect_pending:
        raise ValueError(
            f"pending count mismatch: expected={args.expect_pending} actual={len(pending_rows)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pending_rows)

    payload = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "keyFields": list(KEY_FIELDS),
        "sourceSchedule": {
            "path": str(schedule),
            "sha256": sha256_file(schedule),
            "rows": len(schedule_rows),
        },
        "sourceShards": selected_shards,
        "sourceShardCount": args.source_shard_count,
        "selectedRows": len(selected_rows),
        "completedRowsInSelectedShards": len(selected_keys & completed_keys),
        "pendingRows": len(pending_rows),
        "sourceResults": result_receipts,
        "resumeSchedule": {
            "path": str(output),
            "sha256": sha256_file(output),
            "rows": len(pending_rows),
        },
        "sourceArtifactsModified": False,
    }
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
