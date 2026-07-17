from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def install_cg_package(cg_dir: Path) -> None:
    package = types.ModuleType("cg")
    package.__path__ = [str(cg_dir)]
    package.__package__ = "cg"
    sys.modules["cg"] = package


def load_agent(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path / "main.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load agent from {path}")
    previous = Path.cwd()
    try:
        os.chdir(path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous)
    return module


def play(agent0_dir: Path, agent1_dir: Path, cg_dir: Path, max_decisions: int) -> dict:
    install_cg_package(cg_dir)
    from cg.game import battle_finish, battle_select, battle_start

    agents = [
        load_agent(agent0_dir, "ptcg_agent_0"),
        load_agent(agent1_dir, "ptcg_agent_1"),
    ]
    decks = [
        [int(line) for line in (path / "deck.csv").read_text(encoding="utf-8").splitlines() if line.strip()]
        for path in (agent0_dir, agent1_dir)
    ]
    observation, start = battle_start(decks[0], decks[1])
    if observation is None:
        raise RuntimeError(f"Battle failed to start: player={start.errorPlayer}, type={start.errorType}")

    decisions = 0
    try:
        while decisions < max_decisions:
            current = observation.get("current")
            if current is not None and current.get("result", -1) != -1:
                return {"result": current["result"], "decisions": decisions, "turn": current.get("turn")}
            if current is None:
                player = decisions % 2
            else:
                player = current["yourIndex"]
            action = agents[player].agent(observation)
            observation = battle_select(action)
            decisions += 1
        raise TimeoutError(f"Match exceeded {max_decisions} decisions")
    finally:
        battle_finish()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one match using the official local PTCG engine")
    parser.add_argument("--agent0", default="agents/lucario_rule")
    parser.add_argument("--agent1", default="agents/lucario_rule")
    parser.add_argument("--cg-dir", required=True)
    parser.add_argument("--max-decisions", type=int, default=5000)
    args = parser.parse_args()
    result = play(resolve(args.agent0), resolve(args.agent1), resolve(args.cg_dir), args.max_decisions)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
