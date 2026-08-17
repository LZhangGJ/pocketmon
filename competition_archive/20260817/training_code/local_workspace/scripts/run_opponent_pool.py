from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def run_match(target: Path, opponent: Path, cg_dir: Path, target_seat: int) -> dict:
    agents = [target, opponent] if target_seat == 0 else [opponent, target]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_local_match.py"),
        "--agent0",
        str(agents[0]),
        "--agent1",
        str(agents[1]),
        "--cg-dir",
        str(cg_dir),
        "--max-decisions",
        "5000",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    result["target_seat"] = target_seat
    return result


def evaluate(target: Path, cg_dir: Path, manifest: list[dict], games: int) -> list[dict]:
    rows: list[dict] = []
    for entry in manifest:
        opponent = resolve(entry["agent_dir"])
        if not (opponent / "main.py").is_file() or not (opponent / "deck.csv").is_file():
            raise FileNotFoundError(f"Opponent is not materialized: {entry['name']} at {opponent}")
        wins = losses = draws = decisions = 0
        for game in range(games):
            result = run_match(target, opponent, cg_dir, game % 2)
            decisions += result["decisions"]
            if result["result"] == 2:
                draws += 1
            elif result["result"] == result["target_seat"]:
                wins += 1
            else:
                losses += 1
        row = {
            "opponent": entry["name"],
            "source": entry["source"],
            "games": games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / games,
            "avg_decisions": decisions / games,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an agent against the local public-opponent pool")
    parser.add_argument("--target", default="agents/lucario_rule")
    parser.add_argument("--cg-dir", required=True)
    parser.add_argument("--manifest", default="configs/opponent_pool.json")
    parser.add_argument("--games", type=int, default=20, help="Games per opponent; seats alternate")
    parser.add_argument("--output", help="Optional JSON summary path")
    args = parser.parse_args()
    if args.games < 2:
        raise ValueError("--games must be at least 2 so both seats are tested")

    manifest_path = resolve(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = evaluate(resolve(args.target), resolve(args.cg_dir), manifest, args.games)
    if args.output:
        output = resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
