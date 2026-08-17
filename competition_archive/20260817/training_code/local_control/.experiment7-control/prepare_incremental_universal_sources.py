from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RECENT_DAYS = tuple(f"2026-08-{day:02d}" for day in range(3, 10))
REHEARSAL_DAYS = ("2026-08-01", "2026-08-02")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def dataset_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("datasets")
    if rows is None:
        rows = [payload["dataset"]]
    if not isinstance(rows, list) or not rows:
        raise ValueError("source manifest has no dataset rows")
    return [dict(row) for row in rows]


def validate_dataset(row: dict[str, Any]) -> None:
    for key in ("features", "tokenCache", "sequenceCache", "identityCache", "decisions"):
        path = Path(row[key])
        if not path.exists():
            raise FileNotFoundError(f"dataset artifact is missing: {key}={path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build new-day + recent-7 + older-rehearsal Universal BC sources")
    parser.add_argument("--current-sources", type=Path, required=True)
    parser.add_argument("--older-sources", type=Path, required=True)
    parser.add_argument("--new-sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    current_path = args.current_sources.resolve()
    older_path = args.older_sources.resolve()
    new_path = args.new_sources.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    current = read_json(current_path)
    older = read_json(older_path)
    new = read_json(new_path)
    for payload, path in ((current, current_path), (older, older_path), (new, new_path)):
        if payload.get("kind") != "experiment7_universal_bc":
            raise ValueError(f"not a Universal BC source manifest: {path}")
        if float(payload.get("minGameScoreExclusive", -1)) != 900.0:
            raise ValueError(f"unexpected score filter: {path}")
        if payload.get("policySource") != "winners" or payload.get("moduleVersions") != "all":
            raise ValueError(f"unexpected policy/module contract: {path}")
        if payload["engineCatalog"]["sha256"] != current["engineCatalog"]["sha256"]:
            raise ValueError(f"engine catalog mismatch: {path}")

    current_by_name = {str(row["name"]): row for row in dataset_rows(current)}
    selected: list[dict[str, Any]] = []
    older_row = dataset_rows(older)
    if len(older_row) != 1:
        raise ValueError("older rehearsal manifest must contain exactly one physical dataset")
    selected.append({**older_row[0], "name": "2026-08-01"})
    for name in (
        "2026-08-02",
        "2026-08-03",
        "2026-08-04",
        "2026-08-05",
        "2026-08-06-part-0-of-2",
        "2026-08-06-part-1-of-2",
        "2026-08-07",
        "2026-08-08",
    ):
        if name not in current_by_name:
            raise ValueError(f"current sources are missing required dataset: {name}")
        selected.append(current_by_name[name])
    new_rows = dataset_rows(new)
    if len(new_rows) != 1:
        raise ValueError("new-day manifest must contain exactly one physical dataset")
    selected.append({**new_rows[0], "name": "2026-08-09"})
    names = [str(row["name"]) for row in selected]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate incremental dataset names: {names}")
    for row in selected:
        validate_dataset(row)

    payload = {
        "kind": "experiment7_universal_bc",
        "schemaVersion": 4,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "fixedCommit": current.get("fixedCommit"),
        "referenceRoot": current["referenceRoot"],
        "engineCatalog": current["engineCatalog"],
        "minGameScoreExclusive": 900.0,
        "moduleVersions": "all",
        "policySource": "winners",
        "datasets": selected,
        "sourceManifests": {
            "current": {"path": str(current_path), "sha256": sha256_file(current_path)},
            "olderRehearsal": {"path": str(older_path), "sha256": sha256_file(older_path)},
            "newDay": {"path": str(new_path), "sha256": sha256_file(new_path)},
        },
        "trainingContract": {
            "newDay": "2026-08-09",
            "recentWindowDays": list(RECENT_DAYS),
            "olderRehearsalDays": list(REHEARSAL_DAYS),
            "calendarDays": ["2026-08-01", "2026-08-02", *RECENT_DAYS],
            "physicalShards": len(selected),
            "initialization": "migrate current 7d Universal BC teacher",
            "promotion": "parity, smoke, then full frozen-pool screening before teacher replacement",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    receipt = {
        "path": str(output),
        "sha256": sha256_file(output),
        "calendarDays": payload["trainingContract"]["calendarDays"],
        "physicalShards": len(selected),
    }
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
