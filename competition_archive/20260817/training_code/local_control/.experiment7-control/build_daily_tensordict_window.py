#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from materialize_tensordict_sources import materialize_row


DATE_PATTERN = re.compile(r"^20\d{2}-\d{2}-\d{2}$")
DEFAULT_MANIFEST_PATTERNS = (
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/cache/"
    "*/prepared/universal_training_sources.json",
    "/dataT0/Free/lzhang/pocketmon-runs/replay-refresh-*/cache/"
    "*/prepared/universal_training_sources.json",
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-7d-scoregt900-20260810/daily/"
    "*/prepared/universal_training_sources.json",
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-*/daily/"
    "*/prepared/universal_training_sources.json",
)


def date_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if DATE_PATTERN.fullmatch(part):
            return part
    raise ValueError(f"no YYYY-MM-DD component in {path}")


def load_candidate(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("datasets")
    if rows is None:
        rows = [payload["dataset"]]
    if len(rows) != 1:
        raise ValueError(f"daily manifest must have one dataset: {path}")
    row = dict(rows[0])
    for key in ("features", "decisions", "tokenCache", "sequenceCache", "identityCache"):
        if not Path(row[key]).exists():
            raise FileNotFoundError(f"missing {key}: {row[key]}")
    return payload, row


def discover(patterns: list[str]) -> dict[str, tuple[Path, dict[str, Any], dict[str, Any]]]:
    selected: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    # Earlier patterns have higher priority (daily > refresh > historical).
    for pattern in patterns:
        paths = [Path(value) for value in sorted(glob.glob(pattern))]
        for path in paths:
            try:
                day = date_from_path(path)
                payload, row = load_candidate(path)
            except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
                continue
            selected.setdefault(day, (path.resolve(), payload, row))
    return selected


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the newest immutable ten-day TensorDict memmap window")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--migration-only", action="store_true")
    parser.add_argument("--min-free-gib", type=int, default=256)
    parser.add_argument("--manifest-pattern", action="append", default=[])
    parser.add_argument(
        "--window-root",
        type=Path,
        default=Path("/suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/windows"),
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=Path("/suedata1/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/tensordict-cache"),
    )
    args = parser.parse_args()
    patterns = args.manifest_pattern or list(DEFAULT_MANIFEST_PATTERNS)
    candidates = discover(patterns)
    days = sorted(candidates)[-args.days :]
    if len(days) < args.days:
        raise SystemExit(f"only {len(days)} complete daily manifests; need {args.days}: {days}")
    end = days[-1]
    window_root = args.window_root.resolve()
    window_root.mkdir(parents=True, exist_ok=True)
    window = window_root / end
    final_sources = window / "tensordict-sources.json"
    if final_sources.exists():
        payload = json.loads(final_sources.read_text(encoding="utf-8"))
        recorded = payload.get("tensorStorage", {}).get("calendarDays")
        if recorded == days:
            print(json.dumps({"status": "already_complete", "sources": str(final_sources), "days": days}))
            return
        raise FileExistsError(final_sources)

    base = candidates[end][1]
    rows: list[dict[str, Any]] = []
    source_manifests: list[str] = []
    for day in days:
        path, _, row = candidates[day]
        rows.append({**row, "name": day})
        source_manifests.append(str(path))
    source_payload = {
        "schemaVersion": 6,
        "kind": "experiment7_universal_bc",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "referenceRoot": base["referenceRoot"],
        "engineCatalog": base["engineCatalog"],
        "policySource": "winners",
        "minGameScoreExclusive": 900.0,
        "moduleVersions": "*",
        "datasets": rows,
        "sourceManifests": source_manifests,
        "trainingContract": {
            "calendarDays": days,
            "windowDays": args.days,
            "storage": "TensorDict-style read-only memmap",
            "promotion": "parity, smoke, frozen-pool and key-opponent screening before admission",
        },
    }
    required_bytes = 0
    for row in rows:
        features = Path(row.get("featuresNpz") or row["features"])
        if features.is_file() and features.suffix == ".npz":
            with zipfile.ZipFile(features) as archive:
                required_bytes += sum(info.file_size for info in archive.infolist())
    free_bytes = shutil.disk_usage(window_root).free
    reserve_bytes = args.min_free_gib * 1024**3
    if free_bytes - required_bytes < reserve_bytes:
        raise SystemExit(
            f"WINDOW_SPACE_GUARD_BLOCK free={free_bytes} required={required_bytes} reserve={reserve_bytes}"
        )
    print(
        json.dumps(
            {"stage": "space_guard", "freeBytes": free_bytes, "requiredBytes": required_bytes, "reserveBytes": reserve_bytes}
        ),
        flush=True,
    )
    write_json_atomic(window / "npz-sources.json", source_payload)

    converted: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    store_root = args.store_root.resolve()
    for row in rows:
        updated, receipt = materialize_row(row, store_root)
        converted.append(updated)
        receipts.append(receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    source_payload["datasets"] = converted
    source_payload["tensorStorage"] = {
        "kind": "experiment7_tensordict_memmap",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "calendarDays": days,
        "parity": "shape/dtype plus six deterministic rows per tensor",
        "datasets": receipts,
    }
    write_json_atomic(final_sources, source_payload)
    write_json_atomic(
        window / "SUCCESS.json",
        {"createdAt": datetime.now(timezone.utc).isoformat(), "sources": str(final_sources), "days": days},
    )
    if args.migration_only:
        write_json_atomic(
            window / "MIGRATION_ONLY.json",
            {
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "status": "cache_migration_only_do_not_launch",
                "sources": str(final_sources),
            },
        )
    else:
        write_json_atomic(
            window / "PENDING_A100.json",
            {
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "status": "waiting_for_current_persistent_bc_early_stop",
                "sources": str(final_sources),
                "profiles": ["standard_1m", "large_256x6"],
                "launchRule": "do not duplicate; start only after the matching old-window persistent process exits",
            },
        )
    print(json.dumps({"status": "complete", "sources": str(final_sources), "days": days}))


if __name__ == "__main__":
    main()
