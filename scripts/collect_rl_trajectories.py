from __future__ import annotations

import argparse
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
    try:
        os.chdir(path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(previous)


def install_cg(cg_dir: Path) -> None:
    package = types.ModuleType("cg")
    package.__path__ = [str(cg_dir)]
    package.__package__ = "cg"
    sys.modules["cg"] = package


def play(agent_dirs: list[Path], cg_dir: Path, episode: int, max_decisions: int) -> list[dict]:
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
                    sample["reward"] = 0.0 if winner == 2 else (1.0 if winner == sample["player"] else -1.0)
                return trajectory
            player = step % 2 if current is None else int(current["yourIndex"])
            action = agents[player].agent(observation)
            select = observation.get("select") or {}
            options = select.get("option") or []
            # Multi-card selections are represented by the mean of their option encodings.
            option_vectors = [action_features(option, index) for index, option in enumerate(options)]
            trajectory.append({
                "episode": episode,
                "step": step,
                "player": player,
                "state": state_features(observation),
                "options": option_vectors,
                "chosen": action,
                "reward": 0.0,
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
    args = parser.parse_args()
    rng = random.Random(args.seed)
    target = (ROOT / args.target).resolve()
    pool = json.loads((ROOT / args.pool).read_text(encoding="utf-8"))
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for episode in range(args.episodes):
            opponent = (ROOT / rng.choice(pool)["agent_dir"]).resolve()
            seats = [target, opponent] if episode % 2 == 0 else [opponent, target]
            for sample in play(seats, (ROOT / args.cg_dir).resolve(), episode, args.max_decisions):
                handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
    print(output)


if __name__ == "__main__":
    main()
