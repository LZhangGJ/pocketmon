from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_row(raw: dict[str, Any], line_number: int, path: Path) -> tuple[str, str]:
    if raw.get("schema_version") != 2:
        raise ValueError(f"{path}:{line_number}: schema_version must be 2")
    episode_id = str(raw.get("episode_id") or "")
    source_sha = str(raw.get("source_sha256") or "")
    if not episode_id or len(source_sha) != 64:
        raise ValueError(f"{path}:{line_number}: missing episode/source identity")
    observation = raw.get("observation") or {}
    select = observation.get("select") or {}
    options, action = select.get("option"), raw.get("action")
    if not isinstance(options, list) or not isinstance(action, list):
        raise ValueError(f"{path}:{line_number}: invalid action/options")
    minimum = int(select.get("minCount", 0))
    maximum = int(select.get("maxCount", minimum))
    legal = (
        minimum <= len(action) <= maximum
        and len(action) == len(set(action))
        and all(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(options) for index in action)
    )
    if not legal:
        raise ValueError(f"{path}:{line_number}: illegal recorded action")
    if float(raw.get("policy_weight")) not in (0.0, 1.0) or float(raw.get("value_weight")) != 1.0:
        raise ValueError(f"{path}:{line_number}: invalid policy/value weight")
    return episode_id, source_sha


def merge(
    inputs: list[Path], deck_maps: list[Path], output: Path, deck_output: Path, report_path: Path
) -> dict[str, Any]:
    if not inputs or len(inputs) != len(deck_maps):
        raise ValueError("inputs and deck maps must be non-empty and have equal length")
    inputs = [path.resolve(strict=True) for path in inputs]
    deck_maps = [path.resolve(strict=True) for path in deck_maps]
    output, deck_output, report_path = output.resolve(), deck_output.resolve(), report_path.resolve()
    for path in (output, deck_output, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    for path in (output, deck_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite merged dataset: {path}")
    output_temp = output.with_suffix(output.suffix + ".tmp")
    deck_temp = deck_output.with_suffix(deck_output.suffix + ".tmp")

    episode_origin: dict[str, tuple[int, str]] = {}
    rows_by_input: list[int] = []
    duplicate_rows = policy_rows = value_rows = 0
    dates = Counter()
    try:
        with gzip.open(output_temp, "wt", encoding="utf-8", compresslevel=6) as target:
            for input_index, path in enumerate(inputs):
                written = 0
                with gzip.open(path, "rt", encoding="utf-8") as source:
                    for line_number, line in enumerate(source, 1):
                        raw = json.loads(line)
                        episode_id, source_sha = validate_row(raw, line_number, path)
                        prior = episode_origin.get(episode_id)
                        if prior is None:
                            episode_origin[episode_id] = (input_index, source_sha)
                        elif prior[1] != source_sha:
                            raise ValueError(f"conflicting source SHA for episode {episode_id}")
                        elif prior[0] != input_index:
                            duplicate_rows += 1
                            continue
                        target.write(json.dumps(raw, ensure_ascii=False, separators=(",", ":")) + "\n")
                        written += 1
                        policy_rows += int(float(raw["policy_weight"]) == 1.0)
                        value_rows += 1
                        created = str((raw.get("manifest") or {}).get("create_time") or "")
                        if len(created) >= 10:
                            dates[created[:10]] += 1
                rows_by_input.append(written)

        deck_entries: dict[tuple[str, int], list[int]] = {}
        duplicate_deck_entries = 0
        with gzip.open(deck_temp, "wt", encoding="utf-8", compresslevel=6) as target:
            for path in deck_maps:
                with gzip.open(path, "rt", encoding="utf-8") as source:
                    for line_number, line in enumerate(source, 1):
                        raw = json.loads(line)
                        key = (str(raw.get("episode_id") or ""), raw.get("player"))
                        deck = raw.get("deck")
                        if key[0] not in episode_origin:
                            continue
                        if key[1] not in (0, 1) or not isinstance(deck, list) or len(deck) != 60:
                            raise ValueError(f"{path}:{line_number}: invalid deck sidecar entry")
                        if key in deck_entries:
                            if deck_entries[key] != deck:
                                raise ValueError(f"conflicting deck sidecar entry for {key}")
                            duplicate_deck_entries += 1
                            continue
                        deck_entries[key] = deck
                        target.write(json.dumps(raw, ensure_ascii=False, separators=(",", ":")) + "\n")
        expected_deck_keys = {(episode_id, player) for episode_id in episode_origin for player in (0, 1)}
        missing_decks = sorted(expected_deck_keys - deck_entries.keys())
        if missing_decks:
            raise ValueError(f"merged deck sidecar is missing {len(missing_decks)} episode/player entries")
        output_temp.replace(output)
        deck_temp.replace(deck_output)
    except Exception:
        output_temp.unlink(missing_ok=True)
        deck_temp.unlink(missing_ok=True)
        raise

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in inputs
        ],
        "deck_maps": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in deck_maps
        ],
        "episodes": len(episode_origin),
        "rows": sum(rows_by_input),
        "rows_by_input": rows_by_input,
        "policy_rows": policy_rows,
        "value_rows": value_rows,
        "invalid_actions": 0,
        "duplicate_rows_skipped": duplicate_rows,
        "conflicting_episode_ids": 0,
        "deck_entries": len(deck_entries),
        "duplicate_deck_entries_skipped": duplicate_deck_entries,
        "missing_deck_entries": 0,
        "rows_by_create_date": dict(sorted(dates.items())),
        "output": str(output),
        "output_sha256": sha256(output),
        "output_bytes": output.stat().st_size,
        "deck_output": str(deck_output),
        "deck_output_sha256": sha256(deck_output),
        "deck_output_bytes": deck_output.stat().st_size,
        "gate_passed": True,
    }
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary_report.replace(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge audited daily replay datasets without episode leakage")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--deck-maps", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deck-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.inputs, args.deck_maps, args.output, args.deck_output, args.report), ensure_ascii=False))


if __name__ == "__main__":
    main()
