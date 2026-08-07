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


def install_cg_package(cg_dir: Path) -> None:
    package = types.ModuleType("cg")
    package.__path__ = [str(cg_dir)]
    package.__package__ = "cg"
    sys.modules["cg"] = package


def purge_agent_modules(path: Path) -> None:
    """Remove import names owned by an agent before loading its package.

    Notebook agents are intentionally self-contained and commonly ship top-level
    packages such as ``rl``.  Loading two submissions in one interpreter would
    otherwise make the second agent reuse the first agent's package from
    ``sys.modules``.  Existing objects keep references to their original module
    objects, so removing the import cache here isolates subsequent imports
    without invalidating the first loaded policy.
    """
    owned_names: set[str] = set()
    for child in path.iterdir():
        if child.name.startswith("."):
            continue
        if child.name == "cg":
            # The runner installs the selected official engine before loading
            # agents. Submission archives also bundle that same package, so it
            # must not be mistaken for agent-private state and purged.
            continue
        if child.is_dir() and (child / "__init__.py").is_file():
            owned_names.add(child.name)
        elif child.is_file() and child.suffix == ".py" and child.stem != "main":
            owned_names.add(child.stem)
    for module_name in list(sys.modules):
        if any(module_name == name or module_name.startswith(f"{name}.") for name in owned_names):
            del sys.modules[module_name]


def load_agent(path: Path, module_name: str):
    purge_agent_modules(path)
    spec = importlib.util.spec_from_file_location(module_name, path / "main.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load agent from {path}")
    previous = Path.cwd()
    inserted = str(path) not in sys.path
    try:
        os.chdir(path)
        if inserted:
            sys.path.insert(0, str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if inserted and str(path) in sys.path:
            sys.path.remove(str(path))
        os.chdir(previous)
    return module


def agent_diagnostics(agents: list[object]) -> list[dict]:
    diagnostics = []
    for module in agents:
        callback = getattr(module, "diagnostics", None)
        if not callable(callback):
            diagnostics.append({})
            continue
        try:
            value = callback()
            diagnostics.append(value if isinstance(value, dict) else {"invalid_diagnostics": True})
        except Exception as exc:
            diagnostics.append({"diagnostics_error": f"{type(exc).__name__}: {exc}"})
    return diagnostics


def seed_agent_rng(seed: int, *, seed_loaded_torch: bool = False) -> None:
    """Seed Python-side agent randomness.

    The competition engine's native library does not expose a seed API.  This
    still makes stochastic Python agents reproducible and the returned metadata
    states the remaining limitation explicitly.
    """
    os.environ["POCKETMON_MATCH_SEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    if seed_loaded_torch and "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)


def play(agent0_dir: Path, agent1_dir: Path, cg_dir: Path, max_decisions: int, seed: int = 0) -> dict:
    seed_agent_rng(seed)
    install_cg_package(cg_dir)
    from cg.game import battle_finish, battle_select, battle_start

    agents = [
        load_agent(agent0_dir, "ptcg_agent_0"),
        load_agent(agent1_dir, "ptcg_agent_1"),
    ]
    seed_agent_rng(seed, seed_loaded_torch=True)
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
                return {
                    "result": current["result"],
                    "decisions": decisions,
                    "turn": current.get("turn"),
                    "seed": seed,
                    "engine_seed_controlled": False,
                    "agent_diagnostics": agent_diagnostics(agents),
                }
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = play(resolve(args.agent0), resolve(args.agent1), resolve(args.cg_dir), args.max_decisions, args.seed)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
