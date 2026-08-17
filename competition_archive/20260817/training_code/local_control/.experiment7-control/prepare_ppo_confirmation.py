from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


KEY_FIELDS = ("learner", "opponent", "seed", "learner_seat")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare additive 100-to-300 PPO pair confirmation")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--learners", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-games-per-pair", type=int, default=300)
    parser.add_argument("--expected-completed-per-pair", type=int, default=100)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    outputs = {
        "learners": output / "selected_learners.json",
        "schedule": output / "selected_schedule300.csv",
        "completed": output / "completed_screening100.csv",
        "pending": output / "pending_add200.csv",
        "receipt": output / "confirmation_receipt.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite PPO confirmation preparation outputs")

    selection = load_json(args.selection.resolve())
    selected_names = [str(name) for name in selection["recommendedPair"]["candidates"]]
    if len(selected_names) != 2 or len(set(selected_names)) != 2:
        raise ValueError(f"confirmation requires a distinct recommended pair: {selected_names}")
    learners = load_json(args.learners.resolve())
    metadata = {row["name"]: row for row in learners.get("agents", [])}
    missing = sorted(set(selected_names) - set(metadata))
    if missing:
        raise ValueError(f"selected candidates are absent from learner manifest: {missing}")

    schedule_fields, all_schedule = read_csv(args.schedule.resolve())
    selected_schedule = [row for row in all_schedule if row["learner"] in selected_names]
    schedule_keys = [key(row) for row in selected_schedule]
    if len(schedule_keys) != len(set(schedule_keys)):
        raise ValueError("selected confirmation schedule contains duplicate keys")
    schedule_key_set = set(schedule_keys)

    result_fields: list[str] | None = None
    completed_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    sources = []
    for path in [candidate.resolve() for candidate in args.results]:
        fields, rows = read_csv(path)
        if result_fields is None:
            result_fields = fields
        elif fields != result_fields:
            raise ValueError(f"result schemas differ: {path}")
        selected_rows = [row for row in rows if row["learner"] in selected_names]
        for row in selected_rows:
            row_key = key(row)
            if row_key not in schedule_key_set:
                raise ValueError(f"completed key is absent from 300-game schedule: {row_key}")
            if row_key in completed_by_key:
                raise ValueError(f"duplicate completed key: {row_key}")
            completed_by_key[row_key] = row
        sources.append({"path": str(path), "sha256": sha256_file(path), "selectedRows": len(selected_rows)})
    if result_fields is None:
        raise ValueError("no screening result files were supplied")

    pair_counts: dict[tuple[str, str], list[int]] = {}
    for row in selected_schedule:
        pair_counts.setdefault((row["learner"], row["opponent"]), [0, 0])[0] += 1
    for row in completed_by_key.values():
        pair_counts[(row["learner"], row["opponent"])][1] += 1
    expected = [args.expected_games_per_pair, args.expected_completed_per_pair]
    bad = {f"{learner}::{opponent}": counts for (learner, opponent), counts in pair_counts.items() if counts != expected}
    if bad:
        raise ValueError(f"unexpected confirmation pair coverage: {bad}")

    completed_rows = [completed_by_key[row_key] for row_key in schedule_keys if row_key in completed_by_key]
    pending_rows = [row for row in selected_schedule if key(row) not in completed_by_key]
    selected_payload = {
        **{name: value for name, value in learners.items() if name != "agents"},
        "agents": [metadata[name] for name in selected_names],
        "selectionRule": "recommended complementary pair from complete 100-game screening",
        "sourceSelection": str(args.selection.resolve()),
    }
    outputs["learners"].write_text(json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(outputs["schedule"], schedule_fields, selected_schedule)
    write_csv(outputs["completed"], result_fields, completed_rows)
    write_csv(outputs["pending"], schedule_fields, pending_rows)
    receipt = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "selectedLearners": selected_names,
        "matchups": len(pair_counts),
        "targetRows": len(selected_schedule),
        "completedScreeningRows": len(completed_rows),
        "pendingRows": len(pending_rows),
        "sources": {
            "selection": {"path": str(args.selection.resolve()), "sha256": sha256_file(args.selection.resolve())},
            "learners": {"path": str(args.learners.resolve()), "sha256": sha256_file(args.learners.resolve())},
            "schedule": {"path": str(args.schedule.resolve()), "sha256": sha256_file(args.schedule.resolve())},
            "results": sources,
        },
    }
    outputs["receipt"].write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: receipt[name] for name in ("selectedLearners", "targetRows", "completedScreeningRows", "pendingRows")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
