from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.public_replay import audit_action_positions, merge_position_audits


def replay_files(input_root: Path, date: str | None, max_files: int) -> list[Path]:
    if not input_root.exists():
        return []
    roots = (
        [input_root / date]
        if date
        else [path for path in sorted(input_root.iterdir()) if path.is_dir() and path.name != "_index"]
    )
    files = [path for root in roots if root.exists() for path in sorted(root.glob("*.json"))]
    return files[:max_files] if max_files else files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DATA-001 diagnostic: locate actions relative to ACTIVE replay observations"
    )
    parser.add_argument("--input-root", default="data/raw/replays")
    parser.add_argument("--date")
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--min-lag", type=int, default=-1)
    parser.add_argument("--max-lag", type=int, default=4)
    parser.add_argument("--expected-lag", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--output", default="results/data001_action_positions.json")
    args = parser.parse_args()
    if args.max_lag < args.min_lag:
        raise ValueError("--max-lag must be >= --min-lag")
    lags = tuple(range(args.min_lag, args.max_lag + 1))
    if args.expected_lag not in lags:
        raise ValueError("--expected-lag must be inside the lag range")

    files = replay_files(Path(args.input_root), args.date, args.max_files)
    reports: list[dict] = []
    load_errors: list[dict[str, str]] = []
    for path in files:
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
            reports.append(
                audit_action_positions(
                    replay,
                    lags=lags,
                    expected_lag=args.expected_lag,
                    max_examples=args.max_examples,
                )
            )
        except Exception as exc:
            load_errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    merged = merge_position_audits(
        reports,
        lags=lags,
        expected_lag=args.expected_lag,
        max_examples=args.max_examples,
    )
    report = {
        "experiment_id": "DATA-001-ACTION-POSITION",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(Path(args.input_root)),
        "date": args.date,
        "files_selected": len(files),
        "files_loaded": len(reports),
        "load_errors": load_errors[:20],
        "lag_definition": "action_step - observation_step",
        "framework_expected_lag": 1,
        "framework_semantics": {
            "agent_input": "steps[t-1].observation when steps[t-1].status == ACTIVE",
            "recorded_action": "steps[t].action",
            "recorded_status_beside_action": "post-interpreter steps[t].status",
        },
        "diagnostic_only": True,
        "result": merged,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
