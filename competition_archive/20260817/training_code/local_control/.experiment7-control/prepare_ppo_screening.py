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


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in KEY_FIELDS)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the best smoke candidate per PPO role and prepare an additive 100-game screen"
    )
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--learners", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-games-per-pair", type=int, default=100)
    parser.add_argument("--expected-completed-per-pair", type=int, default=20)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    outputs = {
        "learners": output / "selected_learners.json",
        "schedule": output / "selected_schedule100.csv",
        "completed": output / "completed_smoke20.csv",
        "pending": output / "pending_add80.csv",
        "receipt": output / "screening_receipt.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite PPO screening preparation outputs")

    selection = load_json(args.selection.resolve())
    learners = load_json(args.learners.resolve())
    metadata = {row["name"]: row for row in learners.get("agents", [])}
    selected_names: list[str] = []
    selected_roles: set[str] = set()
    for name in selection.get("individualOrder", []):
        row = metadata.get(name)
        if row is None:
            raise ValueError(f"selection references unknown learner: {name}")
        role = str(row["experiment7Role"])
        if role not in selected_roles:
            selected_names.append(name)
            selected_roles.add(role)
    expected_roles = {str(row["experiment7Role"]) for row in metadata.values()}
    if selected_roles != expected_roles:
        raise ValueError(f"did not select exactly one learner per role: {selected_roles} != {expected_roles}")

    schedule_fields, all_schedule = read_csv(args.schedule.resolve())
    selected_schedule = [row for row in all_schedule if row["learner"] in selected_names]
    schedule_keys = [row_key(row) for row in selected_schedule]
    if len(schedule_keys) != len(set(schedule_keys)):
        raise ValueError("selected schedule contains duplicate match keys")

    result_fields: list[str] | None = None
    completed_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    source_results = []
    schedule_key_set = set(schedule_keys)
    for path in [candidate.resolve() for candidate in args.results]:
        fields, rows = read_csv(path)
        if result_fields is None:
            result_fields = fields
        elif fields != result_fields:
            raise ValueError(f"result schemas differ: {path}")
        selected_rows = [row for row in rows if row["learner"] in selected_names]
        for row in selected_rows:
            key = row_key(row)
            if key not in schedule_key_set:
                raise ValueError(f"completed smoke key is absent from 100-game schedule: {key}")
            if key in completed_by_key:
                raise ValueError(f"duplicate completed smoke key: {key}")
            completed_by_key[key] = row
        source_results.append(
            {"path": str(path), "sha256": sha256_file(path), "selectedRows": len(selected_rows)}
        )
    if result_fields is None:
        raise ValueError("no result files were supplied")

    pair_counts: dict[tuple[str, str], list[int]] = {}
    for row in selected_schedule:
        pair_counts.setdefault((row["learner"], row["opponent"]), [0, 0])[0] += 1
    for row in completed_by_key.values():
        pair_counts[(row["learner"], row["opponent"])][1] += 1
    bad_pairs = {
        f"{learner}::{opponent}": counts
        for (learner, opponent), counts in pair_counts.items()
        if counts != [args.expected_games_per_pair, args.expected_completed_per_pair]
    }
    if bad_pairs:
        raise ValueError(f"unexpected per-pair coverage: {bad_pairs}")

    completed_rows = [completed_by_key[key] for key in schedule_keys if key in completed_by_key]
    pending_rows = [row for row in selected_schedule if row_key(row) not in completed_by_key]
    selected_payload = {
        **{key: value for key, value in learners.items() if key != "agents"},
        "agents": [metadata[name] for name in selected_names],
        "selectionRule": "best zero-failure smoke candidate per Experiment 7 role",
        "sourceSelection": str(args.selection.resolve()),
    }
    outputs["learners"].write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(outputs["schedule"], schedule_fields, selected_schedule)
    write_csv(outputs["completed"], result_fields, completed_rows)
    write_csv(outputs["pending"], schedule_fields, pending_rows)
    receipt = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "selectedLearners": selected_names,
        "selectedRoles": sorted(selected_roles),
        "matchups": len(pair_counts),
        "targetRows": len(selected_schedule),
        "completedSmokeRows": len(completed_rows),
        "pendingRows": len(pending_rows),
        "expected": {
            "learners": 4,
            "opponents": 11,
            "gamesPerPair": args.expected_games_per_pair,
            "completedPerPair": args.expected_completed_per_pair,
        },
        "sources": {
            "selection": {"path": str(args.selection.resolve()), "sha256": sha256_file(args.selection.resolve())},
            "learners": {"path": str(args.learners.resolve()), "sha256": sha256_file(args.learners.resolve())},
            "schedule": {"path": str(args.schedule.resolve()), "sha256": sha256_file(args.schedule.resolve())},
            "results": source_results,
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in outputs.items()
            if name != "receipt"
        },
    }
    outputs["receipt"].write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: receipt[key] for key in ("selectedLearners", "targetRows", "completedSmokeRows", "pendingRows")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
