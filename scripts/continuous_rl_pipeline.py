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
from rl.promotion import (
    build_common_opponent_schedule,
    build_promotion_schedule,
    evaluate_common_opponent_screen,
    evaluate_promotion,
)


PBT_MUTABLE_KEYS = {
    "frozen_league_fraction",
    "rollout_temperature",
    "ppo_epochs",
    "ppo_learning_rate",
    "gamma",
    "gae_lambda",
    "clip_ratio",
    "value_clip",
    "value_coefficient",
    "entropy_coefficient",
    "gradient_clip_norm",
    "target_kl",
}


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


def build_rollout_pool(
    public: list[dict[str, Any]], population_paths: list[str],
) -> list[dict[str, Any]]:
    public_items = [{**item, "league_role": "public"} for item in public]
    population_items = [
        {"name": f"population_{index:02d}", "agent_dir": value, "league_role": "population"}
        for index, value in enumerate(population_paths)
    ]
    if not population_items:
        raise ValueError("frozen PPO league requires at least one population checkpoint")
    return public_items + population_items


def generation_training_config(
    config: dict[str, Any], generation: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply a deterministic PBT hyperparameter mutation for one generation."""

    effective = dict(config)
    variants = list(config.get("pbt_variants") or [])
    if not variants:
        return effective, {"name": "base", "generation": generation, "overrides": {}}
    variant = dict(variants[(generation - 1) % len(variants)])
    name = str(variant.get("name") or f"variant_{(generation - 1) % len(variants):02d}")
    overrides = dict(variant.get("overrides") or {})
    multipliers = dict(variant.get("multipliers") or {})
    unsupported = (set(overrides) | set(multipliers)) - PBT_MUTABLE_KEYS
    if unsupported:
        raise ValueError(f"unsupported PBT mutation keys: {sorted(unsupported)}")
    applied: dict[str, Any] = {}
    for key, multiplier in multipliers.items():
        value = float(effective[key]) * float(multiplier)
        if key == "ppo_epochs":
            value = max(1, round(value))
        effective[key] = value
        applied[key] = value
    for key, value in overrides.items():
        effective[key] = int(value) if key == "ppo_epochs" else float(value)
        applied[key] = effective[key]
    for key in ("frozen_league_fraction", "gamma", "gae_lambda"):
        if key in effective and not 0.0 <= float(effective[key]) <= 1.0:
            raise ValueError(f"PBT mutation {key} must be in [0, 1]")
    for key in (
        "rollout_temperature", "ppo_learning_rate", "clip_ratio", "value_clip",
        "value_coefficient", "gradient_clip_norm", "target_kl",
    ):
        if key in effective and float(effective[key]) <= 0.0:
            raise ValueError(f"PBT mutation {key} must be positive")
    if int(effective.get("ppo_epochs", 0)) <= 0 or float(effective.get("entropy_coefficient", 0.0)) < 0.0:
        raise ValueError("PBT mutation produced invalid PPO epochs or entropy coefficient")
    return effective, {
        "name": name,
        "generation": generation,
        "variant_index": (generation - 1) % len(variants),
        "overrides": applied,
    }


def retain_candidate_in_league(config: dict[str, Any], report: dict[str, Any]) -> bool:
    """Keep safe behavioral variants for training without promoting them."""

    if not bool(config.get("retain_rejected_in_league", False)):
        return False
    if not bool((report.get("checks") or {}).get("zero_failures", False)):
        return False
    public_score = float((report.get("candidate_public") or {}).get("score_rate", 0.0))
    worst_regression = float(report.get("worst_matchup_delta", -1.0))
    return (
        public_score >= float(config.get("league_retention_min_public_score", 0.0))
        and worst_regression >= -float(config.get("league_retention_max_worst_regression", 1.0))
    )


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


def free_gpu(host: str, local_host: str) -> tuple[int, int, int, int]:
    command = [
        "nvidia-smi", "--query-gpu=memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command if host == local_host or host == socket.gethostname() else ["ssh", host, remote_command(command)],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    values = []
    for index, line in enumerate(completed.stdout.splitlines()):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3 and all(field.isdigit() for field in fields):
            values.append((int(fields[0]), int(fields[1]), int(fields[2]), index))
    if not values:
        raise RuntimeError(f"no GPU memory data from {host}")
    # Prefer an idle device over a larger but saturated device. Free ratio and
    # absolute memory break ties between similarly idle GPUs.
    return min(values, key=lambda item: (item[2], -item[0] / item[1], -item[0]))


def torch_cuda_device_count(host: str, config: dict[str, Any]) -> int:
    """Probe CUDA through the exact Python/environment used for training."""

    command = [
        config["python"], "-I", "-c",
        "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)",
    ]
    environment = config.get("host_environment", {}).get(host)
    completed = subprocess.run(
        command if host == config["local_host"] or host == socket.gethostname()
        else ["ssh", host, remote_command(command, environment)],
        capture_output=True,
        text=True,
        timeout=45,
        check=True,
        env=({**os.environ, **(environment or {})} if host == config["local_host"] else None),
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines or not lines[-1].isdigit():
        raise RuntimeError(f"invalid torch CUDA probe from {host}")
    return int(lines[-1])


def choose_trainer(config: dict[str, Any]) -> tuple[str, int, dict[str, Any]]:
    availability: dict[str, Any] = {}
    choices = []
    for host in config["trainer_hosts"]:
        try:
            torch_devices = torch_cuda_device_count(host, config)
            if torch_devices <= 0:
                availability[host] = {"error": "training Python cannot initialize CUDA"}
                continue
            free_memory, total_memory, utilization, gpu = free_gpu(host, config["local_host"])
            if gpu >= torch_devices:
                raise RuntimeError("nvidia-smi device is absent from torch")
            availability[host] = {
                "gpu": gpu,
                "torch_cuda_devices": torch_devices,
                "free_memory_mib": free_memory,
                "total_memory_mib": total_memory,
                "utilization_percent": utilization,
            }
            choices.append((utilization, -free_memory / total_memory, -free_memory, host, gpu))
        except Exception:
            availability[host] = {"error": "GPU query failed"}
    if not choices:
        raise RuntimeError("no trainer host has a visible GPU")
    _, _, _, host, gpu = min(choices)
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


def run_distributed_schedule(
    *,
    config: dict[str, Any],
    root: Path,
    schedule: list[dict[str, Any]],
    learners: list[dict[str, Any]],
    opponents: list[dict[str, Any]],
    label: str,
) -> list[dict[str, Any]]:
    """Run a deterministic, resumable league schedule over all evaluation hosts."""

    root.mkdir(parents=True, exist_ok=True)
    schedule_path = root / "schedule.csv"
    learners_path = root / "learners.json"
    opponents_path = root / "opponents.json"
    write_manifest(learners_path, learners)
    write_manifest(opponents_path, opponents)
    with schedule_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("learner", "opponent", "seed", "learner_seat"))
        writer.writeheader()
        writer.writerows(schedule)
    processes = []
    result_paths = []
    hosts = config["evaluation_hosts"]
    for shard, host in enumerate(hosts):
        output = root / f"results_shard_{shard:02d}.csv"
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
            log_path=root / f"shard_{shard:02d}.log",
            environment=config.get("host_environment", {}).get(host),
        )
        processes.append((f"{label}:{host}:{shard}", process, handle))
    wait_processes(processes)
    wait_for_files(result_paths)
    rows = load_results(result_paths)
    if len(rows) != len(schedule):
        raise RuntimeError(f"{label} incomplete: {len(rows)}/{len(schedule)}")
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
        "q_train": root / "action_q",
        "candidate": root / "candidate_agent",
        "gate": root / "gate",
        "deck": root / "deck_evolution",
    }


def ensure_rollouts(
    *, config: dict[str, Any], state: dict[str, Any], generation: int, paths: dict[str, Path],
) -> list[Path]:
    champion = Path(state["champion_package"])
    public = load_manifest(Path(config["public_opponent_pool"]))
    rollout_pool = paths["root"] / "rollout_pool.json"
    write_manifest(rollout_pool, build_rollout_pool(public, list(state["population"])))
    processes = []
    outputs = []
    shard = 0
    configured_workers = config.get("rollout_workers_per_host", 1)
    counterfactual_enabled = float(config.get("counterfactual_rate", 0.0)) > 0.0
    for host in config["rollout_hosts"]:
        workers = int(configured_workers.get(host, 1) if isinstance(configured_workers, dict) else configured_workers)
        if workers <= 0:
            raise ValueError(f"rollout worker count must be positive for {host}")
        episodes = int(config.get("rollout_episodes_per_worker", config.get("rollout_episodes_per_host", 0)))
        if episodes <= 0:
            raise ValueError("rollout episodes per worker must be positive")
        for _worker in range(workers):
            output = paths["rollouts"] / f"shard_{shard:03d}.jsonl.gz"
            counterfactual_output = paths["rollouts"] / f"shard_{shard:03d}.counterfactual.jsonl.gz"
            outputs.append(output)
            if (
                output.is_file()
                and output.with_name(output.name + ".summary.json").is_file()
                and (not counterfactual_enabled or counterfactual_output.is_file())
            ):
                shard += 1
                continue
            command = [
                config["python"], str(Path(config["code_root"]) / "scripts" / "collect_ppo_rollouts.py"),
                "--checkpoint", str(champion / "checkpoint.pt"),
                "--deck", str(champion / "deck.csv"),
                "--pool", str(rollout_pool),
                "--cg-dir", config["cg_dir"],
                "--episodes", str(episodes),
                "--frozen-league-fraction", str(config.get(
                    "frozen_league_fraction", config.get("self_play_fraction", 0.5)
                )),
                "--temperature", str(config["rollout_temperature"]),
                "--seed", str(config["base_seed"] + generation * 100_000 + shard),
                "--run-id", f"g{generation:05d}-s{shard:03d}",
                "--device", "cpu",
                "--output", str(output),
            ]
            if counterfactual_enabled:
                command.extend([
                    "--counterfactual-output", str(counterfactual_output),
                    "--counterfactual-rate", str(config["counterfactual_rate"]),
                    "--counterfactual-candidates", str(config["counterfactual_candidates"]),
                    "--counterfactual-determinizations", str(config["counterfactual_determinizations"]),
                    "--counterfactual-horizon", str(config["counterfactual_horizon"]),
                ])
            environment = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            environment.update(config.get("host_environment", {}).get(host) or {})
            process, handle = start_process(
                host=host,
                local_host=config["local_host"],
                command=command,
                log_path=paths["rollouts"] / f"shard_{shard:03d}.log",
                environment=environment,
            )
            processes.append((f"rollout:{host}:{shard}", process, handle))
            shard += 1
    wait_processes(processes)
    wait_for_files([
        path
        for output in outputs
        for path in (output, output.with_name(output.name + ".summary.json"))
    ])
    if counterfactual_enabled:
        wait_for_files([
            paths["rollouts"] / f"shard_{index:03d}.counterfactual.jsonl.gz"
            for index in range(shard)
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


def ensure_action_q_checkpoint(
    *, config: dict[str, Any], state: dict[str, Any], generation: int,
    paths: dict[str, Path], actor_checkpoint: Path, events: list[dict[str, Any]],
) -> Path | None:
    if float(config.get("counterfactual_rate", 0.0)) <= 0.0:
        return None
    output = paths["q_train"] / "action_q.pt"
    metrics = paths["q_train"] / "metrics.json"
    if output.is_file() and metrics.is_file():
        return output
    rows = sorted(paths["rollouts"].glob("shard_*.counterfactual.jsonl.gz"))
    if not rows:
        raise RuntimeError("counterfactual action-Q training has no shards")
    trainer, gpu, availability = choose_trainer(config)
    events.append({
        "event": "action_q_trainer_selected", "generation": generation,
        "host": trainer, "gpu": gpu, "free_memory": availability, "time": now(),
    })
    command = [
        config["python"], str(Path(config["code_root"]) / "scripts" / "train_action_q.py"),
        "--rows", *[str(path) for path in rows],
        "--actor-checkpoint", str(actor_checkpoint),
        "--output", str(output),
        "--metrics-output", str(metrics),
        "--epochs", str(config["action_q_epochs"]),
        "--batch-size", str(config["action_q_batch_size"]),
        "--learning-rate", str(config["action_q_learning_rate"]),
        "--heads", str(config["action_q_heads"]),
        "--seed", str(config["base_seed"] + generation * 1000 + 811),
        "--device", "auto",
    ]
    previous = Path(state["champion_package"]) / "action_q.pt"
    if previous.is_file():
        command.extend(["--initialize-from", str(previous)])
    environment = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    environment.update(config.get("host_environment", {}).get(trainer) or {})
    run_process(
        host=trainer,
        local_host=config["local_host"],
        command=command,
        log_path=paths["q_train"] / "train.log",
        environment=environment,
    )
    wait_for_files([output, metrics])
    return output


def ensure_candidate_package(
    *, config: dict[str, Any], state: dict[str, Any], generation: int,
    paths: dict[str, Path], checkpoint: Path, q_checkpoint: Path | None = None,
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
    if q_checkpoint is not None:
        command.extend(["--q-checkpoint", str(q_checkpoint)])
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
    schedule = build_promotion_schedule(
        candidate=candidate_name,
        parent=parent_name,
        public_opponents=public_names,
        games_per_public=config["gate_games_per_public"],
        parent_games=config["gate_parent_games"],
        seed=config["base_seed"] + generation * 100_000,
    )
    rows = run_distributed_schedule(
        config=config,
        root=paths["gate"],
        schedule=schedule,
        learners=learners,
        opponents=opponents,
        label="promotion_gate",
    )
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


def ensure_deck_evolution(
    *,
    config: dict[str, Any],
    state: dict[str, Any],
    generation: int,
    paths: dict[str, Path],
) -> dict[str, Any] | None:
    """Mutate the champion deck, quick-screen variants, then strictly confirm one."""

    every = int(config.get("deck_evolution_every", 0))
    if every <= 0 or generation % every:
        return None
    root = paths["deck"]
    report_path = root / "promotion.json"
    if report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)
    champion = Path(state["champion_package"])
    mutations = root / "mutations"
    mutation_manifest = mutations / "manifest.json"
    if not mutation_manifest.is_file():
        command = [
            config["python"], str(Path(config["code_root"]) / "scripts" / "mutate_legal_decks.py"),
            "--base-deck", str(champion / "deck.csv"),
            "--cards", config["card_database"],
            "--pool-manifest", config["public_opponent_pool"],
            "--count", str(config["deck_candidate_count"]),
            "--max-swaps", str(config["deck_max_swaps"]),
            "--seed", str(config["base_seed"] + generation * 1_000_000 + 701),
            "--output", str(mutations),
        ]
        run_process(
            host=config["local_host"], local_host=config["local_host"], command=command,
            log_path=root / "mutate.log",
        )
        wait_for_files([mutation_manifest])
    mutation_data = json.loads(mutation_manifest.read_text(encoding="utf-8"))
    candidates = mutation_data["candidates"]
    if len(candidates) != int(config["deck_candidate_count"]):
        raise RuntimeError("deck mutation manifest has the wrong candidate count")

    base_name = f"deck_base_g{generation:05d}"
    learner_items = [{"name": base_name, "agent_dir": str(champion)}]
    package_by_name = {base_name: champion}
    for index, item in enumerate(candidates):
        name = f"deck_mutant_g{generation:05d}_{index:03d}"
        package = root / "packages" / f"candidate_{index:03d}"
        if not (package / "agent_manifest.json").is_file():
            command = [
                config["python"], str(Path(config["code_root"]) / "scripts" / "materialize_rl_specialist_agent.py"),
                "--checkpoint", str(champion / "checkpoint.pt"),
                "--deck", str(item["deck"]),
                "--output", str(package),
                "--name", name,
            ]
            champion_q = champion / "action_q.pt"
            if champion_q.is_file():
                command.extend(["--q-checkpoint", str(champion_q)])
            run_process(
                host=config["local_host"], local_host=config["local_host"], command=command,
                log_path=root / "materialize.log",
            )
        learner_items.append({"name": name, "agent_dir": str(package)})
        package_by_name[name] = package

    public_items = load_manifest(Path(config["public_opponent_pool"]))
    public_by_name = {str(item["name"]): item for item in public_items}
    screen_names = [str(name) for name in config["deck_screen_opponents"]]
    missing = sorted(set(screen_names) - public_by_name.keys())
    if missing:
        raise ValueError(f"deck screen opponents are absent from public pool: {missing}")
    screen_opponents = [public_by_name[name] for name in screen_names]
    learner_names = [str(item["name"]) for item in learner_items]
    screen_schedule = build_common_opponent_schedule(
        learners=learner_names,
        opponents=screen_names,
        games_per_opponent=int(config["deck_screen_games_per_opponent"]),
        seed=int(config["base_seed"]) + generation * 1_000_000 + 702,
    )
    screen_rows = run_distributed_schedule(
        config=config,
        root=root / "screen",
        schedule=screen_schedule,
        learners=learner_items,
        opponents=screen_opponents,
        label="deck_screen",
    )
    ranking = evaluate_common_opponent_screen(
        screen_rows, learners=learner_names, opponents=screen_names,
    )
    atomic_json(root / "screen_ranking.json", {"generation": generation, "ranking": ranking})
    baseline = next(row for row in ranking if row["learner"] == base_name)
    eligible = [
        row for row in ranking
        if row["learner"] != base_name
        and row["failures"] == 0
        and row["score_rate"] - baseline["score_rate"] >= float(config["deck_screen_min_delta"])
    ]
    if not eligible:
        report = {
            "promote": False,
            "generation": generation,
            "stage": "screen",
            "reason": "no legal mutant cleared the common-opponent screen",
            "baseline": baseline,
            "ranking": ranking,
            "completed_at": now(),
        }
        atomic_json(report_path, report)
        return report

    selected = eligible[0]
    candidate_name = str(selected["learner"])
    candidate_package = package_by_name[candidate_name]
    parent_name = base_name
    public_names = [str(item["name"]) for item in public_items]
    confirmation_schedule = build_promotion_schedule(
        candidate=candidate_name,
        parent=parent_name,
        public_opponents=public_names,
        games_per_public=int(config["deck_confirmation_games_per_public"]),
        parent_games=int(config["deck_confirmation_parent_games"]),
        seed=int(config["base_seed"]) + generation * 1_000_000 + 703,
    )
    confirmation_rows = run_distributed_schedule(
        config=config,
        root=root / "confirmation",
        schedule=confirmation_schedule,
        learners=[
            {"name": candidate_name, "agent_dir": str(candidate_package)},
            {"name": parent_name, "agent_dir": str(champion)},
        ],
        opponents=public_items + [{"name": parent_name, "agent_dir": str(champion)}],
        label="deck_confirmation",
    )
    confirmation = evaluate_promotion(
        confirmation_rows,
        candidate=candidate_name,
        parent=parent_name,
        public_opponents=public_names,
        min_head_to_head_score=float(config["deck_min_head_to_head_score"]),
        min_head_to_head_wilson=float(config["deck_min_head_to_head_wilson"]),
        min_public_delta=float(config["deck_min_public_delta"]),
        max_worst_matchup_regression=float(config["deck_max_worst_matchup_regression"]),
        max_seat_gap=float(config["deck_max_seat_gap"]),
    )
    report = {
        **confirmation,
        "generation": generation,
        "stage": "confirmation",
        "selected_screen_entry": selected,
        "screen_baseline": baseline,
        "screen_ranking_path": str(root / "screen_ranking.json"),
        "promoted_package": str(candidate_package) if confirmation["promote"] else None,
        "completed_at": now(),
    }
    atomic_json(report_path, report)
    return report


def run_generation(config: dict[str, Any], state: dict[str, Any], run_root: Path) -> dict[str, Any]:
    generation = int(state["generation"]) + 1
    for stale in ("consecutive_failures", "last_error", "last_traceback"):
        state.pop(stale, None)
    paths = generation_paths(run_root, generation)
    paths["root"].mkdir(parents=True, exist_ok=True)
    training_config, mutation = generation_training_config(config, generation)
    atomic_json(paths["root"] / "policy_mutation.json", mutation)
    event_log_path = run_root / "events.jsonl"
    events: list[dict[str, Any]] = []
    state.update({"status": "collecting_rollouts", "active_generation": generation, "updated_at": now()})
    atomic_json(run_root / "state.json", state)
    rollouts = ensure_rollouts(config=training_config, state=state, generation=generation, paths=paths)
    state.update({"status": "training_ppo", "updated_at": now()})
    atomic_json(run_root / "state.json", state)
    checkpoint = ensure_candidate_checkpoint(
        config=training_config, state=state, generation=generation, paths=paths, rollouts=rollouts, log=events
    )
    state.update({"status": "training_action_q", "updated_at": now()})
    atomic_json(run_root / "state.json", state)
    q_checkpoint = ensure_action_q_checkpoint(
        config=training_config, state=state, generation=generation, paths=paths,
        actor_checkpoint=checkpoint, events=events,
    )
    state.update({"status": "packaging_candidate", "updated_at": now()})
    atomic_json(run_root / "state.json", state)
    candidate = ensure_candidate_package(
        config=config, state=state, generation=generation, paths=paths,
        checkpoint=checkpoint, q_checkpoint=q_checkpoint,
    )
    state.update({"status": "promotion_gate", "updated_at": now()})
    atomic_json(run_root / "state.json", state)
    report = ensure_gate(
        config=config, state=state, generation=generation, paths=paths, candidate_package=candidate
    )
    previous = state["champion_package"]
    working_state = dict(state)
    working_population = list(state["population"])
    if report["promote"]:
        working_state["champion_package"] = str(candidate)
        working_state["champion_checkpoint_sha256"] = sha256_file(candidate / "checkpoint.pt")
        working_population.append(str(candidate))
    league_retained = False
    if not report["promote"] and retain_candidate_in_league(config, report):
        working_population.append(str(candidate))
        league_retained = True
    state.update({"status": "deck_evolution", "updated_at": now()})
    atomic_json(run_root / "state.json", state)
    deck_report = ensure_deck_evolution(
        config=config, state=working_state, generation=generation, paths=paths,
    )
    if deck_report and deck_report["promote"]:
        promoted_deck = str(deck_report["promoted_package"])
        working_state["champion_package"] = promoted_deck
        working_state["champion_checkpoint_sha256"] = sha256_file(Path(promoted_deck) / "checkpoint.pt")
        working_population.append(promoted_deck)
    state["champion_package"] = working_state["champion_package"]
    state["champion_checkpoint_sha256"] = working_state["champion_checkpoint_sha256"]
    state["population"] = working_population[-int(config["population_limit"]):]
    state["generation"] = generation
    state.pop("active_generation", None)
    any_promotion = bool(report["promote"] or (deck_report and deck_report["promote"]))
    state["status"] = "promoted" if any_promotion else "rejected"
    state["history"].append({
        "generation": generation,
        "candidate_package": str(candidate),
        "parent_package": previous,
        "promoted": bool(report["promote"]),
        "pbt_variant": mutation,
        "league_retained": league_retained,
        "promotion_report": str(paths["gate"] / "promotion.json"),
        "deck_evolution_run": deck_report is not None,
        "deck_promoted": bool(deck_report and deck_report["promote"]),
        "deck_promotion_report": str(paths["deck"] / "promotion.json") if deck_report else None,
        "final_champion_package": state["champion_package"],
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
