from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
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


def build_rows(files: list[Path], max_files: int | None = None) -> list[dict]:
    rows: list[dict] = []
    file_list = files[: max_files] if max_files else files
    for replay_path in file_list:
        try:
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        info = replay.get("info", {})
        episode_id = info.get("EpisodeId")

        for step_idx, pair in enumerate(replay.get("steps", [])):
            if not isinstance(pair, list):
                continue
            for agent_idx, agent_step in enumerate(pair):
                if not isinstance(agent_step, dict):
                    continue
                action = agent_step.get("action")
                obs = agent_step.get("observation") or {}
                select = obs.get("select") or {}

                if action is None or not isinstance(select, dict):
                    continue

                options = select.get("option")
                option_count = len(options) if isinstance(options, list) else 0

                row = {
                    "episode_id": episode_id,
                    "step_idx": step_idx,
                    "agent_idx": agent_idx,
                    "select_type": select.get("type", -1),
                    "select_context": select.get("context", -1),
                    "option_count": option_count,
                    "min_count": select.get("minCount", -1),
                    "max_count": select.get("maxCount", -1),
                    "target_action": _action_to_text(action),
                }
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare baseline tabular dataset from replay JSON files")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD, default: all available dates")
    parser.add_argument("--max-files", type=int, default=0, help="Limit replay files for quick experiments")
    parser.add_argument("--output", default=str(OUT_CSV))
    args = parser.parse_args()

    replay_files = _iter_replay_files(RAW_REPLAYS, args.date)
    rows = build_rows(replay_files, max_files=(args.max_files or None))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print(f"replay_files={len(replay_files)}")
    print(f"rows={len(df)}")
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
