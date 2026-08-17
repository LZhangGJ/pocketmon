#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload.get("datasets", [])
    receipts = payload.get("tensorStorage", {}).get("datasets", [])
    feature_paths = [Path(row["features"]) for row in rows]
    cache_paths = [
        Path(row[key])
        for row in rows
        for key in ("tokenCache", "sequenceCache", "identityCache")
    ]
    result = {
        "manifest": str(manifest),
        "kind": payload.get("kind"),
        "calendarDays": payload.get("tensorStorage", {}).get("calendarDays", []),
        "datasetCount": len(rows),
        "totalDecisions": sum(int(row.get("summary", {}).get("decisions", 0)) for row in rows),
        "allFeaturesAreMemmapDirectories": all(path.is_dir() for path in feature_paths),
        "allFeatureMetadataPresent": all((path / "meta.json").is_file() for path in feature_paths),
        "allAuxiliaryCachesPresent": all(path.is_dir() for path in cache_paths),
        "allParityPassed": len(receipts) == len(rows) and all(
            receipt.get("parity", {}).get("passed") is True for receipt in receipts
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all(
        result[key]
        for key in (
            "allFeaturesAreMemmapDirectories",
            "allFeatureMetadataPresent",
            "allAuxiliaryCachesPresent",
            "allParityPassed",
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
