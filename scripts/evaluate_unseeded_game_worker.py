from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl.unseeded_eval import (NetworkBlocker, OfficialCabtModuleFinder, approved_terminal,
    install_agent_cg_alias, loaded_native_libraries, outcome_from_rewards, require_sha256)

NATIVE_SHA = "feafd4046b2f688bdb33a4972c139b78e13e243ab5707ece52c43cf39a34b887"


def load_agent(agent_dir: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, agent_dir / "main.py")
    if spec is None or spec.loader is None:
        raise ImportError(agent_dir / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def wrapped(observation):
        previous = Path.cwd()
        try:
            os.chdir(agent_dir)
            return module.agent(observation)
        finally:
            os.chdir(previous)
    return wrapped


def write_result(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    row = {key: spec[key] for key in ("game_id", "schedule_position", "seat0", "seat1")}
    row.update({"phase": "starting", "returned_normally": False, "normal_terminal": False,
                "process_crash": False, "exception": None, "network_attempts": 0})
    write_result(args.result, row)
    started = time.perf_counter()
    try:
        archive_cg = Path(spec["archive_root"]) / "cg"
        sys.path[:0] = [spec["archive_root"], spec["runtime_root"]]
        sys.meta_path.insert(0, OfficialCabtModuleFinder(archive_cg))
        import kaggle_environments
        from kaggle_environments import make
        from kaggle_environments.envs.cabt.cg import game as loaded_game
        from kaggle_environments.envs.cabt.cg import sim as loaded_sim
        native_path = Path(loaded_sim.lib._name).resolve()
        native_hash = require_sha256(native_path, NATIVE_SHA, "loaded native")
        mapped = loaded_native_libraries()
        mapped_hashes = {str(path): require_sha256(path, NATIVE_SHA, "mapped native") for path in mapped}
        expected = {
            "framework": str((Path(spec["runtime_root"]) / "kaggle_environments/__init__.py").resolve()),
            "game": str((archive_cg / "game.py").resolve()), "sim": str((archive_cg / "sim.py").resolve()),
            "native": str((archive_cg / "libcg.so").resolve())}
        actual = {"framework": str(Path(kaggle_environments.__file__).resolve()),
                  "game": str(Path(loaded_game.__file__).resolve()), "sim": str(Path(loaded_sim.__file__).resolve()),
                  "native": str(native_path)}
        if actual != expected or not mapped:
            raise RuntimeError(f"official runtime resolution mismatch: {actual}")
        install_agent_cg_alias(archive_cg, loaded_sim)
        first = load_agent(ROOT / "agents/official_random", f"official_random_first_{spec['game_id']}")
        second = load_agent(ROOT / "agents/official_random", f"official_random_second_{spec['game_id']}")
        by_name = {"official_random_first": first, "official_random_second": second}
        agents = [by_name[spec["seat0"]], by_name[spec["seat1"]]]
        blocker = NetworkBlocker()
        row.update({"phase": "runtime_loaded", "module_paths": actual, "mapped_native_libraries": mapped_hashes,
                    "native_sha256": native_hash, "native_hash_verified": True})
        write_result(args.result, row)
        with blocker:
            env = make("cabt", configuration={"actTimeout": spec["act_timeout"],
                "runTimeout": spec["run_timeout"], "episodeSteps": spec["episode_steps"]}, debug=False)
            env.run(agents)
        final = env.steps[-1]
        statuses = [state.status for state in final]
        rewards = [state.reward for state in final]
        normal = approved_terminal(statuses, rewards, True)
        row.update({"phase": "finished", "returned_normally": True, "normal_terminal": normal,
            "statuses": statuses, "rewards": rewards, "outcome": outcome_from_rewards(rewards),
            "terminal_class": "status_reward_terminal" if normal else None,
            "step_count": len(env.steps), "decision_count": max(0, len(env.steps) - 2),
            "network_attempts": blocker.attempts})
    except Exception as exc:
        row["exception"] = f"{type(exc).__name__}: {exc}"
    row["elapsed_seconds"] = time.perf_counter() - started
    write_result(args.result, row)
    return 0 if row.get("normal_terminal") else 2


if __name__ == "__main__":
    raise SystemExit(main())
