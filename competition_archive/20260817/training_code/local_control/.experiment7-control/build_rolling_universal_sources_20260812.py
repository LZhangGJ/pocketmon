from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    values = payload.get("datasets")
    if values is None:
        values = [payload["dataset"]]
    return [dict(value) for value in values]


def validate_row(row: dict[str, Any]) -> None:
    for key in ("features", "decisions", "tokenCache", "sequenceCache", "identityCache"):
        path = Path(row[key])
        if not path.exists():
            raise FileNotFoundError(f"missing {key}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--rehearsal-08-01", type=Path, required=True)
    parser.add_argument("--new-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    current = read_json(args.current.resolve())
    rehearsal = read_json(args.rehearsal_08_01.resolve())
    by_name = {str(row["name"]): row for row in rows(current)}
    selected: list[dict[str, Any]] = []

    rehearsal_rows = rows(rehearsal)
    if len(rehearsal_rows) != 1:
        raise ValueError("08-01 rehearsal manifest must contain one dataset")
    selected.append({**rehearsal_rows[0], "name": "2026-08-01-rehearsal"})
    selected.append({**by_name["2026-08-02"], "name": "2026-08-02-rehearsal"})
    for name in (
        "2026-08-05",
        "2026-08-06-part-0-of-2",
        "2026-08-06-part-1-of-2",
        "2026-08-07",
        "2026-08-08",
    ):
        selected.append(by_name[name])
    for date in ("2026-08-09", "2026-08-10", "2026-08-11"):
        manifest = read_json(
            args.new_cache_root.resolve() / date / "prepared" / "universal_training_sources.json"
        )
        day_rows = rows(manifest)
        if len(day_rows) != 1:
            raise ValueError(f"{date} manifest must contain one dataset")
        selected.append({**day_rows[0], "name": date})
    for row in selected:
        validate_row(row)

    output = {
        "kind": "experiment7_universal_bc",
        "schemaVersion": 5,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "referenceRoot": current["referenceRoot"],
        "engineCatalog": current["engineCatalog"],
        "minGameScoreExclusive": 900.0,
        "moduleVersions": "all",
        "policySource": "winners",
        "datasets": selected,
        "trainingContract": {
            "latestOfficialDay": "2026-08-11",
            "recentWindowDays": [
                "2026-08-05",
                "2026-08-06",
                "2026-08-07",
                "2026-08-08",
                "2026-08-09",
                "2026-08-10",
                "2026-08-11",
            ],
            "olderRehearsalDays": ["2026-08-01", "2026-08-02"],
            "initialization": "current promoted Universal BC teacher",
            "promotion": "parity, smoke, then frozen-pool screening; do not replace teacher automatically",
            "hashValidation": "omitted by experiment policy",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "datasets": len(selected)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
