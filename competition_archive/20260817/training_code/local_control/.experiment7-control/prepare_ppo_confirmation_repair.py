from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


KEY_FIELDS = ("learner", "opponent", "seed", "learner_seat")
VALID_RESULTS = {"win", "loss", "draw"}


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


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare exact-key repair for failed PPO confirmation rows")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expect-failures", type=int, required=True)
    args = parser.parse_args()

    source = args.source_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    schedule_fields, schedule_rows = read_csv(source / "pending_add200.csv")
    schedule_by_key = {key(row): row for row in schedule_rows}
    if len(schedule_by_key) != len(schedule_rows):
        raise ValueError("pending confirmation schedule contains duplicate keys")

    result_paths = sorted(source.glob("results-add200-shard-*.csv"))
    if len(result_paths) != 12:
        raise ValueError(f"expected 12 source result shards, found {len(result_paths)}")
    result_fields: list[str] | None = None
    result_rows: list[dict[str, str]] = []
    for path in result_paths:
        fields, rows = read_csv(path)
        if result_fields is None:
            result_fields = fields
        elif fields != result_fields:
            raise ValueError(f"result header mismatch: {path}")
        result_rows.extend(rows)
    if result_fields is None:
        raise ValueError("no source results")
    result_by_key = {key(row): row for row in result_rows}
    if len(result_by_key) != len(result_rows):
        raise ValueError("source results contain duplicate keys")
    if set(result_by_key) != set(schedule_by_key):
        raise ValueError("source result keys do not exactly cover pending_add200 schedule")

    failed_keys = {
        row_key
        for row_key, row in result_by_key.items()
        if row.get("failure", "").strip() or row.get("result", "") not in VALID_RESULTS
    }
    if len(failed_keys) != args.expect_failures:
        raise ValueError(
            f"failure count mismatch: expected={args.expect_failures} actual={len(failed_keys)}"
        )
    failed_schedule = [row for row in schedule_rows if key(row) in failed_keys]
    successful_results = [row for row in result_rows if key(row) not in failed_keys]
    write_csv(output / "failed_schedule.csv", schedule_fields, failed_schedule)
    write_csv(output / "completed_add200_success.csv", result_fields, successful_results)
    shutil.copy2(source / "selected_learners.json", output / "selected_learners.json")

    receipt = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sourceRoot": str(source),
        "sourceResultRows": len(result_rows),
        "failedRows": len(failed_schedule),
        "preservedSuccessfulRows": len(successful_results),
        "failedOpponents": sorted({row["opponent"] for row in failed_schedule}),
        "failedLearners": sorted({row["learner"] for row in failed_schedule}),
        "artifacts": {
            name: {
                "path": str(output / name),
                "sha256": sha256_file(output / name),
            }
            for name in (
                "failed_schedule.csv",
                "completed_add200_success.csv",
                "selected_learners.json",
            )
        },
        "sourceArtifactsModified": False,
    }
    with (output / "repair_receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
