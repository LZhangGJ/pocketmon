from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.ppo import sha256_file
from rl.promotion import build_promotion_schedule, evaluate_promotion


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("agents", []) if isinstance(payload, dict) else payload
    return [dict(item) for item in items if item.get("status", "accepted") == "accepted"]


def write_manifest(path: Path, items: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def remote_command(command: list[str], environment: dict[str, str] | None = None) -> str:
    values = []
    if environment:
        values.extend(["env", *[f"{key}={value}" for key, value in sorted(environment.items())]])
    values.extend(command)
    return shlex.join(values)


def start_process(
    *,
    host: str,
    local_host: str,
    command: list[str],
    log_path: Path,
    environment: dict[str, str] | None = None,
) -> tuple[subprocess.Popen, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    if host == local_host or host == socket.gethostname():
        child_environment = os.environ.copy()
        child_environment.update(environment or {})
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=child_environment)
    else:
        process = subprocess.Popen(
            ["ssh", host, remote_command(command, environment)],
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return process, handle


def wait_processes(processes: list[tuple[str, subprocess.Popen, Any]]) -> None:
    failures = []
    for label, process, handle in processes:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append((label, return_code))
    if failures:
        raise RuntimeError(f"pipeline child processes failed: {failures}")


def wait_for_files(paths: list[Path], timeout_seconds: float = 60.0) -> None:
    """Wait for shared-disk metadata to become visible after remote exit."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        missing = [path for path in paths if not path.is_file()]
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"shared-disk artifacts did not become visible: {missing}")
        time.sleep(1.0)


def run_process(
    *,
    host: str,
    local_host: str,
    command: list[str],
    log_path: Path,
    environment: dict[str, str] | None = None,
) -> None:
    process, handle = start_process(
        host=host, local_host=local_host, command=command, log_path=log_path, environment=environment
    )
    wait_processes([(host, process, handle)])


def free_gpu(host: str, local_host: str) -> tuple[int, int]:
    command = [
        "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command if host == local_host or host == socket.gethostname() else ["ssh", host, remote_command(command)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    values = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip().isdigit()]
    if not values:
        raise RuntimeError(f"no GPU memory data from {host}")
    index = max(range(len(values)), key=values.__getitem__)
    return values[index], index


def choose_trainer(config: dict[str, Any]) -> tuple[str, int, dict[str, int]]:
    availability: dict[str, int] = {}
    choices = []
    for host in config["trainer_hosts"]:
        try:
            memory, gpu = free_gpu(host, config["local_host"])
            availability[host] = memory
            choices.append((memory, host, gpu))
        except Exception:
            availability[host] = -1
    if not choices:
        raise RuntimeError("no trainer host has a visible GPU")
    _, host, gpu = max(choices)
    return host, gpu, availability


def load_results(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    seen: set[tuple[str, str, int, int]] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["learner"], row["opponent"], int(row["seed"]), int(row["learner_seat"]))
                if key in seen:
                    raise ValueError(f"duplicate promotion result: {key}")
                seen.add(key)
                rows.append(row)
    return rows


def initial_state(config: dict[str, Any]) -> dict[str, Any]:
    champion = Path(config["initial_champion_package"])
    return {
        "schema_version": 1,
        "status": "initialized",
        "generation": 0,
        "champion_package": str(champion),
        "champion_checkpoint_sha256": sha256_file(champion / "checkpoint.pt"),
        "population": [str(champion)],
        "history": [],
        "updated_at": now(),
    }


def generation_paths(run_root: Path, generation: int) -> dict[str, Path]:
    root = run_root / f"generation_{generation:05d}"
    return {
        "root": root,
        "rollouts": root / "rollouts",
        "train": root / "train",
        "candidate": root / "candidate_agent",
        "gate": root / "gate",
    }


def ensure_rollouts(
    *, config: dict[str, Any], state: dict[str, Any], generation: int, paths: dict[str, Path],
) -> list[Path]:
    champion = Path(state["champion_package"])
    public = load_manifest(Path(config["public_opponent_pool"]))
    population = [
        {"name": f"population_{index:02d}", "agent_dir": value}
        for index, value in enumerate(state["population"])
    ]
    rollout_pool = paths["root"] / "rollout_pool.json"
    write_manifest(rollout_pool, public + population)
    processes = []
    outputs = []
    for shard, host in enumerate(config["rollout_hosts"]):
        output = paths["rollouts"] / f"shard_{shard:02d}.jsonl.gz"
        outputs.append(output)
        if output.is_file() and output.with_name(output.name + ".summary.json").is_file():
            continue
        command = [
            config["python"], str(Path(config["code_root"]) / "scripts" / "collect_ppo_rollouts.py"),
            "--checkpoint", str(champion / "checkpoint.pt"),
            "--deck", str(champion / "deck.csv"),
            "--pool", str(rollout_pool),
            "--cg-dir", config["cg_dir"],
            "--episodes", str(config["rollout_episodes_per_host"]),
            "--self-play-fraction", str(config["self_play_fraction"]),
            "--temperature", str(config["rollout_temperature"]),
            "--seed", str(config["base_seed"] + generation * 1000 + shard),
            "--run-id", f"g{generation:05d}-s{shard:02d}",
            "--device", "cpu",
            "--output", str(output),
        ]
        process, handle = start_process(
            host=host,
            local_host=config["local_host"],
            command=command,
            log_path=paths["rollouts"] / f"shard_{shard:02d}.log",
            environment=config.get("host_environment", {}).get(host),
        )
        processes.append((f"rollout:{host}:{shard}", process, handle))
    wait_processes(processes)
    wait_for_files([
        path
        for output in outputs
        for path in (output, output.with_name(output.name + ".summary.json"))
    ])
    return outputs


def ensure_candidate_checkpoint(
    *, config: dict[str, Any], state: dict[str, Any], generation: int,
    paths: dict[str, Path], rollouts: list[Path], log: list[dict[str, Any]],
) -> Path:
    output = paths["train"] / "candidate.pt"
    metrics = paths["train"] / "metrics.json"
    if output.is_file() and metrics.is_file():
        return output
    trainer, gpu, availability = choose_trainer(config)
    log.append({"event": "trainer_selected", "generation": generation, "host": trainer, "gpu": gpu, "free_memory": availability, "time": now()})
    command = [
        config["python"], str(Path(config["code_root"]) / "scripts" / "train_masked_ppo.py"),
        "--rollouts", *[str(path) for path in rollouts],
        "--initialize-from", str(Path(state["champion_package"]) / "checkpoint.pt"),
        "--output", str(output),
        "--metrics-output", str(metrics),
        "--generation", str(generation),
        "--seed", str(config["base_seed"] + generation),
        "--ppo-epochs", str(config["ppo_epochs"]),
        "--batch-size", str(config["ppo_batch_size"]),
        "--learning-rate", str(config["ppo_learning_rate"]),
        "--gamma", str(config["gamma"]),
        "--gae-lambda", str(config["gae_lambda"]),
        "--clip-ratio", str(config["clip_ratio"]),
        "--value-clip", str(config["value_clip"]),
        "--value-coefficient", str(config["value_coefficient"]),
        "--entropy-coefficient", str(config["entropy_coefficient"]),
        "--gradient-clip-norm", str(config["gradient_clip_norm"]),
        "--target-kl", str(config["target_kl"]),
        "--device", "auto",
    ]
    run_process(
        host=trainer,
        local_host=config["local_host"],
        command=command,
        log_path=paths["train"] / "train.log",
        environment={"CUDA_VISIBLE_DEVICES": str(gpu)},
    )
    wait_for_files([output, metrics])
    return output


def ensure_candidate_package(
    *, config: dict[str, Any], state: dict[str, Any], generation: int,
    paths: dict[str, Path], checkpoint: Path,
) -> Path:
    output = paths["candidate"]
    if (output / "agent_manifest.json").is_file():
        return output
    command = [
        config["python"], str(Path(config["code_root"]) / "scripts" / "materialize_rl_specialist_agent.py"),
        "--checkpoint", str(checkpoint),
        "--deck", str(Path(state["champion_package"]) / "deck.csv"),
        "--output", str(output),
        "--name", f"ppo_candidate_g{generation:05d}",
    ]
    run_process(
        host=config["local_host"], local_host=config["local_host"], command=command,
        log_path=paths["root"] / "materialize.log",
    )
    return output


def ensure_gate(
    *, config: dict[str, Any], state: dict[str, Any], generation: int,
    paths: dict[str, Path], candidate_package: Path,
) -> dict[str, Any]:
    report_path = paths["gate"] / "promotion.json"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    public_items = load_manifest(Path(config["public_opponent_pool"]))
    public_names = [str(item["name"]) for item in public_items]
    candidate_name = f"candidate_g{generation:05d}"
    parent_name = f"parent_g{generation - 1:05d}"
    learners = [
        {"name": candidate_name, "agent_dir": str(candidate_package)},
        {"name": parent_name, "agent_dir": state["champion_package"]},
    ]
    opponents = list(public_items) + [{"name": parent_name, "agent_dir": state["champion_package"]}]
    learners_path = paths["gate"] / "learners.json"
    opponents_path = paths["gate"] / "opponents.json"
    schedule_path = paths["gate"] / "schedule.csv"
    paths["gate"].mkdir(parents=True, exist_ok=True)
    write_manifest(learners_path, learners)
    write_manifest(opponents_path, opponents)
    schedule = build_promotion_schedule(
        candidate=candidate_name,
        parent=parent_name,
        public_opponents=public_names,
        games_per_public=config["gate_games_per_public"],
        parent_games=config["gate_parent_games"],
        seed=config["base_seed"] + generation * 100_000,
    )
    with schedule_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("learner", "opponent", "seed", "learner_seat"))
        writer.writeheader()
        writer.writerows(schedule)
    processes = []
    result_paths = []
    hosts = config["evaluation_hosts"]
    for shard, host in enumerate(hosts):
        output = paths["gate"] / f"results_shard_{shard:02d}.csv"
        result_paths.append(output)
        command = [
            config["python"], str(Path(config["code_root"]) / "scripts" / "run_league_schedule.py"),
            "--schedule", str(schedule_path),
            "--learners", str(learners_path),
            "--opponents", str(opponents_path),
            "--shard-index", str(shard),
            "--shard-count", str(len(hosts)),
            "--output", str(output),
            "--cg-dir", config["cg_dir"],
        ]
        process, handle = start_process(
            host=host,
            local_host=config["local_host"],
            command=command,
            log_path=paths["gate"] / f"shard_{shard:02d}.log",
            environment=config.get("host_environment", {}).get(host),
        )
        processes.append((f"gate:{host}:{shard}", process, handle))
    wait_processes(processes)
    wait_for_files(result_paths)
    rows = load_results(result_paths)
    if len(rows) != len(schedule):
        raise RuntimeError(f"promotion gate incomplete: {len(rows)}/{len(schedule)}")
    report = evaluate_promotion(
        rows,
        candidate=candidate_name,
        parent=parent_name,
        public_opponents=public_names,
        min_head_to_head_score=config["min_head_to_head_score"],
        min_head_to_head_wilson=config["min_head_to_head_wilson"],
        min_public_delta=config["min_public_delta"],
        max_worst_matchup_regression=config["max_worst_matchup_regression"],
        max_seat_gap=config["max_seat_gap"],
    )
    report.update({"generation": generation, "games": len(rows), "completed_at": now()})
    atomic_json(report_path, report)
    return report


def run_generation(config: dict[str, Any], state: dict[str, Any], run_root: Path) -> dict[str, Any]:
    generation = int(state["generation"]) + 1
    paths = generation_paths(run_root, generation)
    paths["root"].mkdir(parents=True, exist_ok=True)
    event_log_path = run_root / "events.jsonl"
    events: list[dict[str, Any]] = []
    state.update({"status": "collecting_rollouts", "active_generation": generation, "updated_at": now()})
    atomic_json(run_root / "state.json", state)
    rollouts = ensure_rollouts(config=config, state=state, generation=generation, paths=paths)
    state.update({"status": "training_ppo", "updated_at": now()})
    atomic_json(run_root / "state.json", state)
    checkpoint = ensure_candidate_checkpoint(
        config=config, state=state, generation=generation, paths=paths, rollouts=rollouts, log=events
    )
    state.update({"status": "packaging_candidate", "updated_at": now()})
    atomic_json(run_root / "state.json", state)
    candidate = ensure_candidate_package(
        config=config, state=state, generation=generation, paths=paths, checkpoint=checkpoint
    )
    state.update({"status": "promotion_gate", "updated_at": now()})
    atomic_json(run_root / "state.json", state)
    report = ensure_gate(
        config=config, state=state, generation=generation, paths=paths, candidate_package=candidate
    )
    previous = state["champion_package"]
    if report["promote"]:
        state["champion_package"] = str(candidate)
        state["champion_checkpoint_sha256"] = sha256_file(candidate / "checkpoint.pt")
        state["population"] = (state["population"] + [str(candidate)])[-int(config["population_limit"]):]
    state["generation"] = generation
    state.pop("active_generation", None)
    state["status"] = "promoted" if report["promote"] else "rejected"
    state["history"].append({
        "generation": generation,
        "candidate_package": str(candidate),
        "parent_package": previous,
        "promoted": bool(report["promote"]),
        "promotion_report": str(paths["gate"] / "promotion.json"),
        "completed_at": now(),
    })
    state["updated_at"] = now()
    atomic_json(run_root / "state.json", state)
    if events:
        with event_log_path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent population self-play PPO and promotion coordinator")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if os.name == "nt":
        raise RuntimeError("continuous coordinator must run on a Linux training server")
    import fcntl

    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_root = Path(config["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (run_root / "pipeline.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another continuous RL coordinator already holds the lock") from exc
    state_path = run_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else initial_state(config)
    atomic_json(state_path, state)
    failures = 0
    while not (run_root / "STOP").exists():
        max_generations = int(config.get("max_generations", 0))
        if max_generations and int(state["generation"]) >= max_generations:
            state.update({"status": "complete", "updated_at": now()})
            atomic_json(state_path, state)
            break
        try:
            state = run_generation(config, state, run_root)
            failures = 0
            time.sleep(float(config.get("sleep_between_generations_seconds", 5)))
        except Exception as exc:
            failures += 1
            state.update({
                "status": "error",
                "consecutive_failures": failures,
                "last_error": f"{type(exc).__name__}: {exc}",
                "last_traceback": traceback.format_exc(),
                "updated_at": now(),
            })
            atomic_json(state_path, state)
            if failures >= int(config.get("max_consecutive_failures", 3)):
                state["status"] = "blocked"
                atomic_json(state_path, state)
                break
            time.sleep(float(config.get("failure_backoff_seconds", 60)))
    if (run_root / "STOP").exists():
        state.update({"status": "stopped", "updated_at": now()})
        atomic_json(state_path, state)


if __name__ == "__main__":
    main()
