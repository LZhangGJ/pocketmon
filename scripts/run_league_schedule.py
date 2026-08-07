from __future__ import annotations

import argparse
import csv
import json
import py_compile
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_FIELDS = [
    "learner",
    "opponent",
    "seed",
    "learner_seat",
    "result",
    "winner_seat",
    "decisions",
    "latency_ms",
    "memory_mb",
    "failure",
    "engine_seed_controlled",
    "diagnostics_json",
]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_agents(path: Path) -> dict[str, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("agents", []) if isinstance(payload, dict) else payload
    agents: dict[str, Path] = {}
    for item in items:
        if item.get("status", "accepted") != "accepted":
            continue
        agent_dir = item.get("agent_dir") or item.get("path")
        if not agent_dir:
            raise ValueError(f"agent has no path: {item}")
        agents[str(item["name"])] = resolve(agent_dir)
    return agents


def row_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["learner"]),
        str(row["opponent"]),
        int(row["seed"]),
        int(row["learner_seat"]),
    )


def validate_agent(path: Path, name: str) -> None:
    missing = [filename for filename in ("main.py", "deck.csv") if not (path / filename).is_file()]
    if missing:
        raise FileNotFoundError(f"{name} missing {', '.join(missing)} at {path}")
    py_compile.compile(str(path / "main.py"), doraise=True)


def classify_result(payload: dict[str, Any], learner_seat: int) -> str:
    winner = int(payload["result"])
    if winner == 2:
        return "draw"
    return "win" if winner == learner_seat else "loss"


def run_match(
    learner: Path,
    opponent: Path,
    cg_dir: Path,
    seed: int,
    learner_seat: int,
    max_decisions: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    agents = [learner, opponent] if learner_seat == 0 else [opponent, learner]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_local_match.py"),
        "--agent0",
        str(agents[0]),
        "--agent1",
        str(agents[1]),
        "--cg-dir",
        str(cg_dir),
        "--seed",
        str(seed),
        "--max-decisions",
        str(max_decisions),
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        return {
            "result": classify_result(payload, learner_seat),
            "winner_seat": payload.get("result", ""),
            "decisions": payload.get("decisions", ""),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "memory_mb": "",
            "failure": "",
            "engine_seed_controlled": bool(payload.get("engine_seed_controlled", False)),
            "diagnostics_json": json.dumps(payload.get("agent_diagnostics", []), separators=(",", ":")),
        }
    except subprocess.TimeoutExpired as exc:
        failure = f"TimeoutExpired: {exc}"
        result = "timeout"
    except (subprocess.CalledProcessError, json.JSONDecodeError, IndexError, KeyError, ValueError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        failure = f"{type(exc).__name__}: {exc}; stderr={stderr[-1000:]}"
        lowered = failure.lower()
        result = "illegal" if "indexerror" in lowered or "illegal" in lowered else "crash"
    return {
        "result": result,
        "winner_seat": "",
        "decisions": "",
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "memory_mb": "",
        "failure": failure,
        "engine_seed_controlled": False,
        "diagnostics_json": "[]",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a League-v1 CSV schedule with resumable per-game logging")
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--learners", required=True, help="JSON manifest mapping learner names to agent directories")
    parser.add_argument("--opponents", default="configs/opponent_pool.json")
    parser.add_argument("--cg-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-decisions", type=int, default=5000)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be in [0, shard count)")

    learners = load_agents(resolve(args.learners))
    opponents = load_agents(resolve(args.opponents))
    cg_dir = resolve(args.cg_dir)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed_keys: set[tuple[str, str, int, int]] = set()
    if output.is_file():
        with output.open(newline="", encoding="utf-8") as handle:
            completed_keys = {row_key(row) for row in csv.DictReader(handle)}

    with resolve(args.schedule).open(newline="", encoding="utf-8") as handle:
        schedule = list(csv.DictReader(handle))
    scheduled = [row for index, row in enumerate(schedule) if index % args.shard_count == args.shard_index]
    pending = [row for row in scheduled if row_key(row) not in completed_keys]
    mode = "a" if output.is_file() and output.stat().st_size else "w"
    with output.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if mode == "w":
            writer.writeheader()
        for index, row in enumerate(pending, start=1):
            learner_name = row["learner"]
            opponent_name = row["opponent"]
            if learner_name not in learners:
                raise KeyError(f"unknown learner: {learner_name}")
            if opponent_name not in opponents:
                raise KeyError(f"unknown opponent: {opponent_name}")
            validate_agent(learners[learner_name], learner_name)
            validate_agent(opponents[opponent_name], opponent_name)
            result = run_match(
                learners[learner_name],
                opponents[opponent_name],
                cg_dir,
                int(row["seed"]),
                int(row["learner_seat"]),
                args.max_decisions,
                args.timeout_seconds,
            )
            writer.writerow({**row, **result})
            handle.flush()
            print(json.dumps({"progress": f"{index}/{len(pending)}", **row, **result}, ensure_ascii=False), flush=True)

    print(json.dumps({
        "output": str(output),
        "shard": args.shard_index,
        "shards": args.shard_count,
        "scheduled": len(scheduled),
        "already_completed": len(scheduled) - len(pending),
        "executed": len(pending),
        "engine_seed_controlled": False,
    }))


if __name__ == "__main__":
    main()
