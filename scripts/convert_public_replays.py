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


REPORT_COUNTERS = (
    "episodes",
    "rows",
    "policy_rows",
    "value_rows",
    "empty_action_rows",
    "invalid_decisions",
    "setup_actions",
    "initial_actions_skipped",
    "non_acting_actions_skipped",
    "unknown_submission_status_skipped",
    "episodes_terminal_reward",
    "episodes_terminal_forfeit",
    "episodes_result_fallback",
    "episodes_unresolved_terminal_reward",
    "episodes_missing_winner",
    "parsed_rows",
    "quarantined_episodes",
    "quarantined_rows",
    "quarantined_reward_mismatches",
    "quarantined_missing_winners",
    "top_level_reward_episodes",
    "reward_mismatches",
    "duplicate_episode_ids",
    "conflicting_episode_ids",
    "load_errors",
)


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
    parser.add_argument(
        "--quarantine-outcome-errors",
        action="store_true",
        help="Exclude episodes with missing or contradictory outcome labels instead of assigning a winner",
    )
    parser.add_argument("--max-quarantined-episode-rate", type=float, default=0.0)
    parser.add_argument("--output", default="data/processed/public_replay_v1.jsonl.gz")
    parser.add_argument("--report", default="results/data002_public_replay_conversion.json")
    args = parser.parse_args()
    if not 0.0 <= args.max_invalid_rate <= 1.0:
        raise ValueError("--max-invalid-rate must be in [0, 1]")
    if not 0.0 <= args.max_quarantined_episode_rate <= 1.0:
        raise ValueError("--max-quarantined-episode-rate must be in [0, 1]")

    files = replay_files(Path(args.input_root), args.date, args.max_files)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    manifests: dict[Path, dict[str, dict[str, str]]] = {}
    seen: dict[str, str] = {}
    counters: Counter[str] = Counter()
    for key in REPORT_COUNTERS:
        counters[key] = 0
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
                    counters["parsed_rows"] += len(rows)
                    counters["invalid_decisions"] += episode_report["invalid_decisions"]
                    counters["setup_actions"] += episode_report["setup_actions"]
                    counters["initial_actions_skipped"] += episode_report.get("initial_actions_skipped", 0)
                    counters["non_acting_actions_skipped"] += episode_report.get("non_acting_actions_skipped", 0)
                    counters["unknown_submission_status_skipped"] += episode_report.get(
                        "unknown_submission_status_skipped", 0
                    )
                    winner_source = episode_report.get("winner_source")
                    if winner_source == "terminal_reward":
                        counters["episodes_terminal_reward"] += 1
                    elif winner_source == "terminal_forfeit":
                        counters["episodes_terminal_forfeit"] += 1
                    elif winner_source == "observation_result":
                        counters["episodes_result_fallback"] += 1
                    elif winner_source == "unresolved_terminal_reward":
                        counters["episodes_unresolved_terminal_reward"] += 1
                    if episode_report.get("top_level_reward_present"):
                        counters["top_level_reward_episodes"] += 1
                    if episode_report.get("reward_mismatch"):
                        counters["reward_mismatches"] += 1
                        errors.append(
                            {
                                "path": str(path),
                                "episode_id": episode_id,
                                "reason": "terminal rewards disagree with top-level rewards",
                            }
                        )
                    if episode_report["winner"] is None:
                        counters["episodes_missing_winner"] += 1
                        errors.append({"path": str(path), "episode_id": episode_id, "reason": "terminal winner was not found"})
                    if episode_report["invalid_decisions"]:
                        errors.append({"path": str(path), **episode_report})
                    outcome_error = bool(
                        episode_report.get("reward_mismatch") or episode_report["winner"] is None
                    )
                    quarantine = args.quarantine_outcome_errors and outcome_error
                    if quarantine:
                        counters["quarantined_episodes"] += 1
                        counters["quarantined_rows"] += len(rows)
                        counters["quarantined_reward_mismatches"] += int(
                            bool(episode_report.get("reward_mismatch"))
                        )
                        counters["quarantined_missing_winners"] += int(episode_report["winner"] is None)
                    else:
                        counters["rows"] += len(rows)
                        counters["policy_rows"] += sum(row["policy_weight"] > 0 for row in rows)
                        counters["value_rows"] += sum(row["value_weight"] > 0 for row in rows)
                        counters["empty_action_rows"] += sum(row["action"] == [] for row in rows)
                        for row in rows:
                            handle.write(json_dumps(row) + "\n")
                except Exception as exc:
                    counters["load_errors"] += 1
                    errors.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})

        decision_total = counters["rows"] + counters["invalid_decisions"]
        invalid_rate = counters["invalid_decisions"] / decision_total if decision_total else 1.0
        quarantined_episode_rate = (
            counters["quarantined_episodes"] / counters["episodes"] if counters["episodes"] else 1.0
        )
        outcome_gate_passed = (
            counters["quarantined_reward_mismatches"] == counters["reward_mismatches"]
            and counters["quarantined_missing_winners"] == counters["episodes_missing_winner"]
            and quarantined_episode_rate <= args.max_quarantined_episode_rate
        ) if args.quarantine_outcome_errors else (
            counters["episodes_missing_winner"] == 0 and counters["reward_mismatches"] == 0
        )
        gate_passed = (
            bool(counters["rows"])
            and invalid_rate <= args.max_invalid_rate
            and counters["load_errors"] == 0
            and counters["conflicting_episode_ids"] == 0
            and outcome_gate_passed
            and counters["unknown_submission_status_skipped"] == 0
            and (args.policy_source == "all" or counters["policy_rows"] > 0)
        )
        report = {
            "experiment_id": "DATA-002",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "alignment": args.alignment,
            "policy_source": args.policy_source,
            "keep_logs": args.keep_logs,
            "quarantine_outcome_errors": args.quarantine_outcome_errors,
            "files_selected": len(files),
            **dict(counters),
            "invalid_rate": invalid_rate,
            "max_invalid_rate": args.max_invalid_rate,
            "quarantined_episode_rate": quarantined_episode_rate,
            "max_quarantined_episode_rate": args.max_quarantined_episode_rate,
            "outcome_gate_passed": outcome_gate_passed,
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
