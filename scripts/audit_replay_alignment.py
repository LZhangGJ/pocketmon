from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.public_replay import audit_replay, merge_audits


def replay_files(input_root: Path, date: str | None, max_files: int) -> list[Path]:
    roots = [input_root / date] if date else [path for path in sorted(input_root.iterdir()) if path.is_dir() and path.name != "_index"]
    files = [path for root in roots if root.exists() for path in sorted(root.glob("*.json"))]
    return files[:max_files] if max_files else files


def main() -> None:
    parser = argparse.ArgumentParser(description="DATA-001: determine public replay action/observation alignment")
    parser.add_argument("--input-root", default="data/raw/replays")
    parser.add_argument("--date")
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--output", default="results/data001_replay_alignment.json")
    parser.add_argument("--min-valid-rate", type=float, default=0.999)
    parser.add_argument("--strict", action="store_true", help="Return exit code 2 unless the recommended alignment clears the gate")
    args = parser.parse_args()

    files = replay_files(Path(args.input_root), args.date, args.max_files)
    reports = {"previous": [], "same": []}
    load_errors: list[dict[str, str]] = []
    for path in files:
        try:
            replay = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            load_errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
            continue
        for alignment in reports:
            reports[alignment].append(audit_replay(replay, alignment))

    merged = {alignment: merge_audits(values, alignment) for alignment, values in reports.items()}
    recommended = max(merged, key=lambda name: (merged[name]["valid_rate"], merged[name]["valid_decisions"])) if files else None
    gate_passed = bool(recommended) and merged[recommended]["valid_rate"] >= args.min_valid_rate and not load_errors
    report = {
        "experiment_id": "DATA-001",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(Path(args.input_root)),
        "date": args.date,
        "files_selected": len(files),
        "files_loaded": len(files) - len(load_errors),
        "load_errors": load_errors[:20],
        "alignments": merged,
        "recommended_alignment": recommended,
        "min_valid_rate": args.min_valid_rate,
        "gate_passed": gate_passed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
