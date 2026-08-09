from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.public_replay import iter_transitions, terminal_outcome, validate_transition


def load_agent(agent_dir: Path) -> Any:
    sys.path.insert(0, str(agent_dir))
    spec = importlib.util.spec_from_file_location(
        "submitted_55328694_main", agent_dir / "main.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load submitted agent")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata_lookup(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["id"]): row for row in manifest.get("episodes") or []}


def our_players(metadata: dict[str, Any], submission_id: int) -> list[int]:
    return sorted(
        int(agent["index"])
        for agent in metadata.get("agents") or []
        if int(agent.get("submission_id", -1)) == submission_id
    )


def reset_stats(module: Any) -> None:
    module.HISTORY.clear()
    for key in module._STATS:
        module._STATS[key] = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_dir = args.input.resolve()
    manifest = json.loads((input_dir / "episodes.json").read_text(encoding="utf-8"))
    by_id = metadata_lookup(manifest)
    submitted = load_agent(args.agent_dir.resolve())

    rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    paths = sorted((input_dir / "replays").glob("*.json"))
    for position, path in enumerate(paths, start=1):
        replay = json.loads(path.read_text(encoding="utf-8"))
        episode_id = int((replay.get("info") or {}).get("EpisodeId"))
        metadata = by_id[episode_id]
        winner = terminal_outcome(replay)["winner"]
        for player in our_players(metadata, args.submission_id):
            reset_stats(submitted)
            comparisons = 0
            mismatches: list[dict[str, Any]] = []
            for transition in iter_transitions(replay, "previous"):
                if transition.player != player or transition.observation is None:
                    continue
                validation = validate_transition(transition)
                if not validation.valid:
                    continue
                predicted = submitted.agent(transition.observation)
                comparisons += 1
                if predicted != transition.action:
                    mismatches.append(
                        {
                            "action_step": transition.action_step,
                            "kind": validation.kind,
                            "recorded": transition.action,
                            "predicted": predicted,
                        }
                    )
            stats = submitted.bc_advisor.get_stats()
            result = "draw" if winner == 2 else "win" if winner == player else "loss"
            row = {
                "episode_id": episode_id,
                "episode_type": metadata.get("type"),
                "player": player,
                "result": result,
                "comparisons": comparisons,
                "mismatch_count": len(mismatches),
                "mismatch_examples": mismatches[:10],
                "stats": stats,
            }
            rows.append(row)
            totals["player_episodes"] += 1
            totals["comparisons"] += comparisons
            totals["mismatches"] += len(mismatches)
            for key, value in stats.items():
                totals[key] += int(value)
        print(f"replays={position}/{len(paths)} id={episode_id}", flush=True)

    output = {
        "submission_id": args.submission_id,
        "replay_files": len(paths),
        "totals": dict(totals),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output["totals"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
