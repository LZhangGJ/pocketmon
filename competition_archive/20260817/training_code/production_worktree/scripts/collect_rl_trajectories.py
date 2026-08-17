from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import random
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.features import action_features, state_features


def load_agent(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path / "main.py")
    if spec is None or spec.loader is None:
        raise ImportError(path)
    previous = Path.cwd()
    inserted = str(path) not in sys.path
    try:
        os.chdir(path)
        if inserted:
            sys.path.insert(0, str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted and str(path) in sys.path:
            sys.path.remove(str(path))
        os.chdir(previous)


def install_cg(cg_dir: Path) -> None:
    package = types.ModuleType("cg")
    package.__path__ = [str(cg_dir)]
    package.__package__ = "cg"
    sys.modules["cg"] = package


def compact_observation(observation: dict) -> dict:
    return {
        "current": observation.get("current"),
        "select": observation.get("select"),
    }


def play(agent_dirs: list[Path], cg_dir: Path, episode: int, episode_id: str, max_decisions: int) -> list[dict]:
    install_cg(cg_dir)
    from cg.game import battle_finish, battle_select, battle_start

    agents = [load_agent(path, f"rl_teacher_{episode}_{seat}") for seat, path in enumerate(agent_dirs)]
    decks = [[int(line) for line in (path / "deck.csv").read_text().splitlines() if line] for path in agent_dirs]
    observation, start = battle_start(*decks)
    if observation is None:
        raise RuntimeError(f"battle start failed: {start.errorType}")
    trajectory: list[dict] = []
    try:
        for step in range(max_decisions):
            current = observation.get("current")
            if current is not None and current.get("result", -1) != -1:
                winner = current["result"]
                for sample in trajectory:
                    outcome = 0.0 if winner == 2 else (1.0 if winner == sample["player"] else -1.0)
                    sample["winner"] = winner
                    sample["outcome"] = outcome
                    sample["reward"] = outcome
                    sample["policy_weight"] = float(outcome > 0)
                    sample["value_weight"] = 1.0
                return trajectory
            player = step % 2 if current is None else int(current["yourIndex"])
            action = agents[player].agent(observation)
            select = observation.get("select") or {}
            options = select.get("option") or []
            if not isinstance(select, dict) or not isinstance(options, list):
                observation = battle_select(action)
                continue
            trajectory.append({
                "schema_version": 2,
                "episode_id": episode_id,
                "episode": episode,
                "action_step": step + 1,
                "observation_step": step,
                "player": player,
                "observation": compact_observation(observation),
                "action": action,
                "chosen": action,
                "reward": 0.0,
                "source_path": "local_agent_league",
                "source_sha256": episode_id,
                "action_status": "legal_engine_accepted",
            })
            observation = battle_select(action)
        raise TimeoutError(f"episode {episode} exceeded {max_decisions} decisions")
    finally:
        battle_finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect teacher/self-play trajectories for offline RL")
    parser.add_argument("--cg-dir", required=True)
    parser.add_argument("--target", default="agents/lucario_rule")
    parser.add_argument("--pool", default="configs/opponent_pool.json")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-decisions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output", default="data/rl/trajectories.jsonl")
    parser.add_argument("--deck-map-output", help="Optional schema-v1 deck sidecar for structured training")
    parser.add_argument("--run-id", default="local-league", help="Unique prefix so independently collected shards can be merged")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    target = (ROOT / args.target).resolve()
    pool = [
        item for item in json.loads((ROOT / args.pool).read_text(encoding="utf-8"))
        if item.get("status", "accepted") == "accepted"
    ]
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(output).endswith(".gz") else open
    deck_output = (ROOT / args.deck_map_output).resolve() if args.deck_map_output else None
    if deck_output is not None:
        deck_output.parent.mkdir(parents=True, exist_ok=True)
    deck_handle = (
        (gzip.open if str(deck_output).endswith(".gz") else open)(deck_output, "wt", encoding="utf-8")
        if deck_output else None
    )
    with opener(output, "wt", encoding="utf-8") as handle:
        for episode in range(args.episodes):
            episode_id = f"{args.run_id}-{episode}"
            opponent = (ROOT / rng.choice(pool)["agent_dir"]).resolve()
            seats = [target, opponent] if episode % 2 == 0 else [opponent, target]
            if deck_handle is not None:
                for player, path in enumerate(seats):
                    deck = [int(line) for line in (path / "deck.csv").read_text().splitlines() if line]
                    deck_handle.write(json.dumps({
                        "schema_version": 1,
                        "episode_id": episode_id,
                        "player": player,
                        "deck": deck,
                        "source_path": str(path),
                    }, separators=(",", ":")) + "\n")
            for sample in play(seats, (ROOT / args.cg_dir).resolve(), episode, episode_id, args.max_decisions):
                handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
    if deck_handle is not None:
        deck_handle.close()
    print(output)


if __name__ == "__main__":
    main()
