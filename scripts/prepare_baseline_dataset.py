from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.public_replay import canonical_rows, replay_sha256

RAW_REPLAYS = ROOT / "data" / "raw" / "replays"
OUT_CSV = ROOT / "data" / "processed" / "baseline_train_rows.csv"


def _iter_replay_files(base: Path, date: str | None) -> list[Path]:
    if date:
        replay_dir = base / date
        return sorted(replay_dir.glob("*.json"))

    files: list[Path] = []
    for child in sorted(base.iterdir()):
        if child.is_dir() and child.name != "_index":
            files.extend(sorted(child.glob("*.json")))
    return files


def _action_to_text(action: object) -> str:
    return json.dumps(action, separators=(",", ":"), ensure_ascii=False)


def build_rows(
    files: list[Path],
    max_files: int | None = None,
    policy_source: str = "winners",
) -> list[dict]:
    """Build a diagnostic table from validated, previous-step-aligned decisions."""

    rows: list[dict] = []
    file_list = files[: max_files] if max_files else files
    for replay_path in file_list:
        try:
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            canonical, report = canonical_rows(
                replay,
                alignment="previous",
                source_path=str(replay_path),
                source_sha256=replay_sha256(replay_path),
                policy_source=policy_source,
            )
        except Exception:
            continue
        if report["invalid_decisions"]:
            raise ValueError(
                f"Replay alignment produced {report['invalid_decisions']} invalid decisions: {replay_path}. "
                "Run scripts/audit_replay_alignment.py before preparing labels."
            )

        for sample in canonical:
            if sample["policy_weight"] <= 0:
                continue
            select = sample["observation"]["select"]
            options = select.get("option") or []
            rows.append(
                {
                    "episode_id": sample["episode_id"],
                    "action_step": sample["action_step"],
                    "observation_step": sample["observation_step"],
                    "agent_idx": sample["player"],
                    "outcome": sample["outcome"],
                    "select_type": select.get("type", -1),
                    "select_context": select.get("context", -1),
                    "option_count": len(options),
                    "min_count": select.get("minCount", -1),
                    "max_count": select.get("maxCount", -1),
                    "target_action": _action_to_text(sample["action"]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a validated diagnostic table from public replay JSON")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD, default: all available dates")
    parser.add_argument("--max-files", type=int, default=0, help="Limit replay files for quick experiments")
    parser.add_argument("--policy-source", choices=("winners", "all"), default="winners")
    parser.add_argument("--output", default=str(OUT_CSV))
    args = parser.parse_args()

    replay_files = _iter_replay_files(RAW_REPLAYS, args.date)
    rows = build_rows(replay_files, max_files=(args.max_files or None), policy_source=args.policy_source)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"replay_files={len(replay_files)}")
    print(f"rows={len(rows)}")
    print("alignment=previous")
    print(f"policy_source={args.policy_source}")
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
