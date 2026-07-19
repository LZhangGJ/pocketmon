from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.public_replay import canonical_rows, json_dumps, replay_episode_id, replay_sha256


def replay_files(input_root: Path, date: str | None, max_files: int) -> list[Path]:
    roots = [input_root / date] if date else [path for path in sorted(input_root.iterdir()) if path.is_dir() and path.name != "_index"]
    files = [path for root in roots if root.exists() for path in sorted(root.glob("*.json"))]
    return files[:max_files] if max_files else files


def manifest_rows(directory: Path) -> dict[str, dict[str, str]]:
    path = directory / "manifest.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        episode_id = next((row.get(key) for key in ("episode_id", "EpisodeId", "id") if row.get(key)), None)
        if episode_id is not None:
            output[str(episode_id)] = row
    return output


def open_text(path: Path, mode: str, *, compressed: bool) -> TextIO:
    if compressed:
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="DATA-002: convert validated public replays to canonical offline-RL JSONL")
    parser.add_argument("--input-root", default="data/raw/replays")
    parser.add_argument("--date")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--alignment", choices=("previous", "same"), default="previous")
    parser.add_argument("--policy-source", choices=("winners", "all"), default="winners")
    parser.add_argument("--keep-logs", action="store_true", help="Keep observation.logs; forbidden for model features")
    parser.add_argument("--max-invalid-rate", type=float, default=0.0)
    parser.add_argument("--output", default="data/processed/public_replay_v1.jsonl.gz")
    parser.add_argument("--report", default="results/data002_public_replay_conversion.json")
    args = parser.parse_args()
    if not 0.0 <= args.max_invalid_rate <= 1.0:
        raise ValueError("--max-invalid-rate must be in [0, 1]")

    files = replay_files(Path(args.input_root), args.date, args.max_files)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    manifests: dict[Path, dict[str, dict[str, str]]] = {}
    seen: dict[str, str] = {}
    counters: Counter[str] = Counter()
    errors: list[dict[str, object]] = []

    try:
        with open_text(temporary, "wt", compressed=str(output).endswith(".gz")) as handle:
            for path in files:
                try:
                    digest = replay_sha256(path)
                    replay = json.loads(path.read_text(encoding="utf-8"))
                    episode_id = replay_episode_id(replay, path.stem)
                    if episode_id in seen:
                        if seen[episode_id] != digest:
                            counters["conflicting_episode_ids"] += 1
                            errors.append({"path": str(path), "episode_id": episode_id, "reason": "same id has different content hash"})
                        else:
                            counters["duplicate_episode_ids"] += 1
                        continue
                    seen[episode_id] = digest
                    manifest = manifests.setdefault(path.parent, manifest_rows(path.parent)).get(episode_id, {})
                    rows, episode_report = canonical_rows(
                        replay,
                        alignment=args.alignment,
                        source_path=str(path),
                        source_sha256=digest,
                        manifest=manifest,
                        policy_source=args.policy_source,
                        keep_logs=args.keep_logs,
                    )
                    counters["episodes"] += 1
                    counters["rows"] += len(rows)
                    counters["policy_rows"] += sum(row["policy_weight"] > 0 for row in rows)
                    counters["value_rows"] += sum(row["value_weight"] > 0 for row in rows)
                    counters["invalid_decisions"] += episode_report["invalid_decisions"]
                    counters["setup_actions"] += episode_report["setup_actions"]
                    if episode_report["invalid_decisions"]:
                        errors.append({"path": str(path), **episode_report})
                    for row in rows:
                        handle.write(json_dumps(row) + "\n")
                except Exception as exc:
                    counters["load_errors"] += 1
                    errors.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})

        decision_total = counters["rows"] + counters["invalid_decisions"]
        invalid_rate = counters["invalid_decisions"] / decision_total if decision_total else 1.0
        gate_passed = (
            bool(counters["rows"])
            and invalid_rate <= args.max_invalid_rate
            and counters["load_errors"] == 0
            and counters["conflicting_episode_ids"] == 0
        )
        report = {
            "experiment_id": "DATA-002",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "alignment": args.alignment,
            "policy_source": args.policy_source,
            "keep_logs": args.keep_logs,
            "files_selected": len(files),
            **dict(counters),
            "invalid_rate": invalid_rate,
            "max_invalid_rate": args.max_invalid_rate,
            "gate_passed": gate_passed,
            "output": str(output),
            "errors": errors[:50],
        }
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not gate_passed:
            raise RuntimeError("DATA-002 conversion gate failed; temporary output was not promoted")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    main()
