from __future__ import annotations

import argparse
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.unseeded_eval import (
    NetworkBlocker,
    OfficialCabtModuleFinder,
    alternating_schedule,
    approved_terminal,
    loaded_native_libraries,
    install_agent_cg_alias,
    outcome_from_rewards,
    require_sha256,
    summarize_stage_a,
)


ARCHIVE_SHA = "09ad210b15476f5064c1509addb32a459c777d92d4e4e7db470f9d0c039c3282"
API_SHA = "593f1298e52a635f90f8f505a52113e9af114f444c293404e37906f18ee06ced"
GAME_SHA = "3bd3d4f4a369a11e6d2f5da9094cf15ebc410a2221835e6417b7cff4883f1fc2"
SIM_SHA = "1555f57f5d22bf4c09d70e0e667a916e575e68c9dd1de9ead34ba5e7e4968655"
NATIVE_SHA = "feafd4046b2f688bdb33a4972c139b78e13e243ab5707ece52c43cf39a34b887"
WHEEL_SHA = "359226741a04fbe1dbbc10121aef140fd96ec4fa31bace2037d05e7ef2bbf4e8"
RUNTIME_NATIVE_SHA = "7acbfc7bc61d4f8233515c63debcfa454b8f804f138a6c395c599decc3dd17d0"
CHECKPOINT_SHA = "2faac94de9e937dee77cd6d5d44036d7f45bb2dc4cc6491c1c97c0091f4fb216"


def git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    return commit, dirty


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

    wrapped.module = module
    return wrapped


def diagnostics(agent) -> dict[str, Any]:
    callback = getattr(agent.module, "diagnostics", None)
    if not callable(callback):
        return {}
    try:
        value = callback()
        return value if isinstance(value, dict) else {"invalid_diagnostics": True}
    except Exception as exc:
        return {"diagnostics_error": f"{type(exc).__name__}: {exc}"}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def current_turn_and_result(final_states) -> tuple[int | None, int | None]:
    for state in reversed(final_states):
        current = (state.observation or {}).get("current")
        if isinstance(current, dict):
            return current.get("turn"), current.get("result")
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description="EVAL-UNSEEDED-001 Stage A runtime gate")
    parser.add_argument("--archive-zip", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--runtime-wheel", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--act-timeout", type=int, default=5)
    parser.add_argument("--run-timeout", type=int, default=120)
    parser.add_argument("--episode-steps", type=int, default=100000)
    parser.add_argument("--gate-output", type=Path, required=True)
    parser.add_argument("--games-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    if args.games != 20:
        raise SystemExit("Stage A requires exactly 20 games")
    for output in (args.gate_output, args.games_output, args.summary_output):
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing evidence: {output}")
    commit, dirty = git_state()
    if dirty:
        raise SystemExit("formal Stage A requires a clean worktree")

    archive_root = args.archive_root.resolve()
    archive_cg = archive_root / "cg"
    runtime_root = args.runtime_root.resolve()
    hashes = {
        "archive_zip": require_sha256(args.archive_zip, ARCHIVE_SHA, "competition archive"),
        "archive_api": require_sha256(archive_cg / "api.py", API_SHA, "competition api.py"),
        "archive_game": require_sha256(archive_cg / "game.py", GAME_SHA, "competition game.py"),
        "archive_sim": require_sha256(archive_cg / "sim.py", SIM_SHA, "competition sim.py"),
        "archive_native": require_sha256(archive_cg / "libcg.so", NATIVE_SHA, "competition libcg.so"),
        "runtime_wheel": require_sha256(args.runtime_wheel, WHEEL_SHA, "kaggle-environments wheel"),
        "runtime_bundled_native": require_sha256(
            runtime_root / "kaggle_environments/envs/cabt/cg/libcg.so",
            RUNTIME_NATIVE_SHA,
            "runtime bundled libcg.so",
        ),
        "checkpoint": require_sha256(args.checkpoint, CHECKPOINT_SHA, "frozen checkpoint"),
    }

    sys.path.insert(0, str(archive_root))
    sys.path.insert(0, str(runtime_root))
    finder = OfficialCabtModuleFinder(archive_cg)
    sys.meta_path.insert(0, finder)
    import kaggle_environments
    from kaggle_environments import make
    from kaggle_environments.envs.cabt.cg import game as loaded_game
    from kaggle_environments.envs.cabt.cg import sim as loaded_sim

    native_path = Path(loaded_sim.lib._name).resolve()
    native_hash = require_sha256(native_path, NATIVE_SHA, "actually loaded CABT native library")
    mapped_libraries = loaded_native_libraries()
    if not mapped_libraries:
        raise RuntimeError("no loaded libcg.so found in /proc/self/maps")
    mapped_hashes = {str(path): require_sha256(path, NATIVE_SHA, "mapped libcg.so") for path in mapped_libraries}
    module_paths = {
        "framework": str(Path(kaggle_environments.__file__).resolve()),
        "game": str(Path(loaded_game.__file__).resolve()),
        "sim": str(Path(loaded_sim.__file__).resolve()),
        "native": str(native_path),
    }
    expected_modules = {
        "framework": str((runtime_root / "kaggle_environments/__init__.py").resolve()),
        "game": str((archive_cg / "game.py").resolve()),
        "sim": str((archive_cg / "sim.py").resolve()),
        "native": str((archive_cg / "libcg.so").resolve()),
    }
    if module_paths != expected_modules:
        raise RuntimeError(f"official wrapper resolution mismatch: {module_paths}")
    install_agent_cg_alias(archive_cg, loaded_sim)

    gate = {
        "experiment_id": "EVAL-UNSEEDED-001",
        "stage": "A",
        "status": "running",
        "code_commit": commit,
        "dirty_at_start": dirty,
        "hashes": hashes,
        "artifact_paths": {
            "archive_zip": str(args.archive_zip.resolve()),
            "archive_root": str(archive_root),
            "runtime_wheel": str(args.runtime_wheel.resolve()),
            "runtime_root": str(runtime_root),
            "checkpoint": str(args.checkpoint.resolve()),
        },
        "module_paths": module_paths,
        "mapped_native_libraries": mapped_hashes,
        "agent_cg_sim_reuses_framework_sim": sys.modules.get("cg.sim") is loaded_sim,
        "loaded_native_sha256": native_hash,
        "host": os.uname().nodename,
        "python": sys.version.replace("\n", " "),
        "commands": [[sys.executable, *sys.argv]],
        "nominal_python_seed_used_as_engine_seed": False,
        "pairing_key_used": False,
    }
    schedule = alternating_schedule(args.games)
    agent_dir = ROOT / "agents/official_random"
    first = load_agent(agent_dir, "eval_unseeded_official_random_first")
    second = load_agent(agent_dir, "eval_unseeded_official_random_second")
    agents_by_name = {"official_random_first": first, "official_random_second": second}
    agent_modules = {
        name: str(Path(agent.module.__file__).resolve()) for name, agent in agents_by_name.items()
    }
    gate["agent_modules"] = agent_modules
    write_json(args.gate_output, gate)
    records = []
    started = time.perf_counter()
    blocker = NetworkBlocker()
    with blocker:
        for item in schedule:
            game_started = time.perf_counter()
            row: dict[str, Any] = {
                **item,
                "agent_identities": [item["seat0"], item["seat1"]],
                "statuses": [],
                "rewards": [],
                "outcome": None,
                "terminal_class": None,
                "turn": None,
                "stale_current_result": None,
                "step_count": 0,
                "decision_count": 0,
                "elapsed_seconds": 0.0,
                "diagnostics": [],
                "returned_normally": False,
                "normal_terminal": False,
                "process_crash": False,
                "exception": None,
                "module_paths": module_paths,
                "agent_modules": agent_modules,
                "native_sha256": native_hash,
                "native_hash_verified": True,
                "hashes": hashes,
            }
            agents = [agents_by_name[item["seat0"]], agents_by_name[item["seat1"]]]
            try:
                env = make(
                    "cabt",
                    configuration={
                        "actTimeout": args.act_timeout,
                        "runTimeout": args.run_timeout,
                        "episodeSteps": args.episode_steps,
                    },
                    debug=False,
                )
                env.run(agents)
                row["returned_normally"] = True
                final = env.steps[-1]
                row["statuses"] = [state.status for state in final]
                row["rewards"] = [state.reward for state in final]
                row["step_count"] = len(env.steps)
                row["decision_count"] = max(0, len(env.steps) - 2)
                row["turn"], row["stale_current_result"] = current_turn_and_result(final)
                row["outcome"] = outcome_from_rewards(row["rewards"])
                row["normal_terminal"] = approved_terminal(
                    row["statuses"], row["rewards"], row["returned_normally"]
                )
                if row["normal_terminal"]:
                    row["terminal_class"] = "status_reward_terminal"
            except Exception as exc:
                row["exception"] = f"{type(exc).__name__}: {exc}"
            row["elapsed_seconds"] = time.perf_counter() - game_started
            row["diagnostics"] = [diagnostics(agent) for agent in agents]
            records.append(row)
            append_jsonl(args.games_output, row)

    summary = summarize_stage_a(records, args.games, blocker.attempts)
    summary.update({
        "experiment_id": "EVAL-UNSEEDED-001",
        "stage": "A",
        "code_commit": commit,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "peak_vram_mb": 0,
        "model_games": 0,
        "stage_b_authorized": False,
        "hashes": hashes,
        "module_paths": module_paths,
        "mapped_native_libraries": mapped_hashes,
        "schedule": schedule,
    })
    summary["stage_b_authorized"] = summary["gate_passed"]
    gate["status"] = "passed" if summary["gate_passed"] else "failed"
    gate["summary"] = summary
    write_json(args.gate_output, gate)
    write_json(args.summary_output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
