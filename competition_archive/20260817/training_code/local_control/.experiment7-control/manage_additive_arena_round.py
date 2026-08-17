from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_learners(path: Path) -> set[str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    names = {str(row["name"]) for row in payload.get("agents", [])}
    if not names:
        raise ValueError(f"empty learners manifest: {path}")
    return names


def prepare(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    schedule_path = args.schedule.resolve()
    learners_path = args.learners.resolve()
    learners = load_learners(learners_path)
    schedule_fields, schedule = read_csv(schedule_path)
    if {row["learner"] for row in schedule} != learners:
        raise ValueError("schedule and learner manifest names differ")
    candidate_keys = [key(row) for row in schedule]
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError("candidate schedule contains duplicate keys")

    result_fields: list[str] | None = None
    completed: dict[tuple[str, ...], dict[str, str]] = {}
    sources = []
    for candidate in args.source_results:
        path = candidate.resolve()
        fields, rows = read_csv(path)
        if result_fields is None:
            result_fields = fields
        elif fields != result_fields:
            raise ValueError(f"source result headers differ: {path}")
        selected = [row for row in rows if row["learner"] in learners]
        for row in selected:
            row_key = key(row)
            if row_key in completed:
                raise ValueError(f"duplicate source result key: {row_key}")
            completed[row_key] = row
        sources.append({"path": str(path), "sha256": sha256_file(path), "selectedRows": len(selected)})
    if result_fields is None:
        raise ValueError("no source results")

    candidate_by_pair: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in schedule:
        candidate_by_pair.setdefault((row["learner"], row["opponent"]), []).append(row)
    completed_by_pair: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in completed.values():
        completed_by_pair.setdefault((row["learner"], row["opponent"]), []).append(row)
    if set(candidate_by_pair) != set(completed_by_pair):
        raise ValueError("candidate schedule and completed source results cover different matchups")

    completed_rows: list[dict[str, str]] = []
    pending_rows: list[dict[str, str]] = []
    target_schedule: list[dict[str, str]] = []
    pair_counts: dict[tuple[str, str], list[int]] = {}
    for pair in sorted(candidate_by_pair):
        pair_completed = completed_by_pair[pair]
        if len(pair_completed) != args.completed_games:
            raise ValueError(f"unexpected completed coverage for {pair}: {len(pair_completed)}")
        completed_keys = {key(row) for row in pair_completed}
        available = [row for row in candidate_by_pair[pair] if key(row) not in completed_keys]
        add_count = args.target_games - args.completed_games
        if len(available) < add_count:
            raise ValueError(f"not enough fresh candidate keys for {pair}: need={add_count} have={len(available)}")
        selected_new = available[:add_count]
        completed_rows.extend(pair_completed)
        pending_rows.extend(selected_new)
        target_schedule.extend(
            {field: row[field] for field in schedule_fields} for row in pair_completed
        )
        target_schedule.extend(selected_new)
        pair_counts[pair] = [len(pair_completed) + len(selected_new), len(pair_completed)]
    target_keys = [key(row) for row in target_schedule]
    if len(target_keys) != len(set(target_keys)):
        raise ValueError("constructed additive target schedule contains duplicate keys")

    add_games = args.target_games - args.completed_games
    selected_name = f"selected_schedule{args.target_games}.csv"
    completed_name = f"completed_round{args.completed_games}.csv"
    pending_name = f"pending_add{add_games}.csv"
    write_csv(output / selected_name, schedule_fields, target_schedule)
    write_csv(output / completed_name, result_fields, completed_rows)
    write_csv(output / pending_name, schedule_fields, pending_rows)
    receipt = {
        "schemaVersion": 2,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "learners": len(learners),
        "matchups": len(pair_counts),
        "candidateRows": len(schedule),
        "targetRows": len(target_schedule),
        "completedRows": len(completed_rows),
        "pendingRows": len(pending_rows),
        "expected": {"targetGamesPerPair": args.target_games, "completedGamesPerPair": args.completed_games},
        "outputs": {
            "selectedSchedule": selected_name,
            "completedResults": completed_name,
            "pendingSchedule": pending_name,
        },
        "sources": {
            "candidateSchedule": {"path": str(schedule_path), "sha256": sha256_file(schedule_path)},
            "learners": {"path": str(learners_path), "sha256": sha256_file(learners_path)},
            "results": sources,
        },
    }
    (output / "additive_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))


def finalize(args: argparse.Namespace) -> None:
    schedule_fields, schedule = read_csv(args.schedule.resolve())
    del schedule_fields
    schedule_keys = [key(row) for row in schedule]
    result_fields: list[str] | None = None
    results: dict[tuple[str, ...], dict[str, str]] = {}
    for candidate in [args.completed.resolve(), *[path.resolve() for path in args.new_results]]:
        fields, rows = read_csv(candidate)
        if result_fields is None:
            result_fields = fields
        elif fields != result_fields:
            raise ValueError(f"final result headers differ: {candidate}")
        for row in rows:
            row_key = key(row)
            if row_key in results:
                raise ValueError(f"duplicate final result key: {row_key}")
            results[row_key] = row
    if set(results) != set(schedule_keys):
        raise ValueError(f"final results do not cover schedule: results={len(results)} schedule={len(schedule_keys)}")
    failures = [row for row in results.values() if row.get("failure", "").strip() or row.get("result", "") not in VALID_RESULTS]
    if failures:
        raise ValueError(f"final additive Arena results contain {len(failures)} failures")
    if result_fields is None:
        raise ValueError("no final results")
    ordered = [results[row_key] for row_key in schedule_keys]
    write_csv(args.output.resolve(), result_fields, ordered)
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(ordered), "failures": 0}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or finalize an additive Arena round")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--schedule", type=Path, required=True)
    prep.add_argument("--learners", type=Path, required=True)
    prep.add_argument("--source-results", type=Path, nargs="+", required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--target-games", type=int, required=True)
    prep.add_argument("--completed-games", type=int, required=True)
    prep.set_defaults(func=prepare)
    final = subparsers.add_parser("finalize")
    final.add_argument("--schedule", type=Path, required=True)
    final.add_argument("--completed", type=Path, required=True)
    final.add_argument("--new-results", type=Path, nargs="+", required=True)
    final.add_argument("--output", type=Path, required=True)
    final.set_defaults(func=finalize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
