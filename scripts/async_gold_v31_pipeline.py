from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.ppo import sha256_file
from scripts.continuous_rl_pipeline import (
    atomic_json,
    ensure_action_q_checkpoint,
    ensure_candidate_checkpoint,
    ensure_candidate_package,
    ensure_rollouts,
    ensure_search_distilled_checkpoint,
    ensure_staged_gate,
    generation_paths,
    generation_training_config,
    load_manifest,
    now,
    q_materialization_arguments,
    retain_candidate_in_league,
    run_process,
)


ACTIVE_EVALUATION_STATUSES = {"queued", "evaluating"}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    base_path = Path(payload["base_config"])
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    config = json.loads(base_path.read_text(encoding="utf-8"))
    config = deep_merge(config, dict(payload.get("overrides") or {}))
    async_config = dict(payload.get("async") or {})
    required = {
        "run_root", "code_root", "python", "local_host", "initial_champion_package",
        "public_opponent_pool", "base_seed", "population_limit", "gate_stages",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"async config is missing required fields: {missing}")
    targets = [int(stage.get("target_games", -1)) for stage in config["gate_stages"]]
    if targets != [20, 200, 400]:
        raise ValueError("async evaluator requires exact 20, 200 and 400 game stages")
    if int(async_config.get("max_pending_candidates", 0)) <= 0:
        raise ValueError("async.max_pending_candidates must be positive")
    return config, async_config


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**payload, "time": now()}, ensure_ascii=False) + "\n")


def acquire_role_lock(run_root: Path, role: str):
    import fcntl

    handle = (run_root / f"{role}.lock").open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"another async {role} already holds the lock") from exc
    return handle


def package_fingerprint(package: Path) -> dict[str, Any]:
    checkpoint = package / "checkpoint.pt"
    deck = package / "deck.csv"
    if not checkpoint.is_file() or not deck.is_file():
        raise FileNotFoundError(f"incomplete candidate package: {package}")
    parts = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "deck_sha256": sha256_file(deck),
        "action_q_sha256": sha256_file(package / "action_q.pt") if (package / "action_q.pt").is_file() else None,
    }
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    parts["candidate_sha256"] = hashlib.sha256(canonical).hexdigest()
    return parts


def initialize_registries(config: dict[str, Any], run_root: Path) -> None:
    champion_path = run_root / "champion.json"
    league_path = run_root / "league.json"
    package = Path(config["initial_champion_package"])
    fingerprint = package_fingerprint(package)
    if not champion_path.is_file():
        atomic_json(champion_path, {
            "schema_version": 1,
            "version": 0,
            "package": str(package),
            **fingerprint,
            "source": "initialization",
            "updated_at": now(),
        })
    if not league_path.is_file():
        atomic_json(league_path, {
            "schema_version": 1,
            "packages": [str(package)],
            "updated_at": now(),
        })


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_roots(run_root: Path) -> list[Path]:
    return sorted(path for path in run_root.glob("generation_*") if path.is_dir())


def lifecycle(root: Path) -> dict[str, Any]:
    path = root / "lifecycle.json"
    return read_json(path) if path.is_file() else {"status": "producing"}


def count_pending(run_root: Path) -> int:
    return sum(lifecycle(root).get("status") in ACTIVE_EVALUATION_STATUSES for root in candidate_roots(run_root))


def candidate_priority(manifest: dict[str, Any], life: dict[str, Any], champion_version: int) -> tuple[Any, ...]:
    current_parent = int(manifest["training_parent_version"]) == int(champion_version)
    return (
        0 if life.get("status") == "evaluating" else 1,
        0 if current_parent else 1,
        -int(manifest.get("priority", 0)),
        str(manifest.get("queued_at", "")),
        int(manifest["generation"]),
    )


def select_candidate(run_root: Path, champion_version: int) -> tuple[Path, dict[str, Any], dict[str, Any]] | None:
    choices = []
    for root in candidate_roots(run_root):
        manifest_path = root / "candidate.json"
        if not manifest_path.is_file():
            continue
        life = lifecycle(root)
        if life.get("status") not in ACTIVE_EVALUATION_STATUSES:
            continue
        manifest = read_json(manifest_path)
        choices.append((candidate_priority(manifest, life, champion_version), root, manifest, life))
    if not choices:
        return None
    _, root, manifest, life = min(choices, key=lambda item: item[0])
    return root, manifest, life


def choose_candidate_kind(generation: int, async_config: dict[str, Any]) -> str:
    every = int(async_config.get("deck_candidate_every", 0))
    return "deck" if every > 0 and generation % every == 0 else "policy"


def producer_state(run_root: Path) -> dict[str, Any]:
    path = run_root / "producer_state.json"
    if path.is_file():
        return read_json(path)
    return {
        "schema_version": 1,
        "status": "initialized",
        "next_generation": 1,
        "active_generation": None,
        "produced": 0,
        "updated_at": now(),
    }


def production_snapshot(run_root: Path, generation: int, async_config: dict[str, Any]) -> dict[str, Any]:
    root = generation_paths(run_root, generation)["root"]
    working_path = root / "working.json"
    if working_path.is_file():
        return read_json(working_path)
    champion = read_json(run_root / "champion.json")
    league = read_json(run_root / "league.json")
    working = {
        "schema_version": 1,
        "generation": generation,
        "kind": choose_candidate_kind(generation, async_config),
        "training_parent_version": int(champion["version"]),
        "training_parent_package": champion["package"],
        "training_parent_checkpoint_sha256": champion["checkpoint_sha256"],
        "population": list(league["packages"]),
        "started_at": now(),
    }
    atomic_json(working_path, working)
    return working


def generation_state(working: dict[str, Any]) -> dict[str, Any]:
    return {
        "generation": int(working["generation"]) - 1,
        "champion_package": working["training_parent_package"],
        "champion_checkpoint_sha256": working["training_parent_checkpoint_sha256"],
        "population": list(working["population"]),
        "history": [],
    }


def produce_policy_candidate(
    config: dict[str, Any], run_root: Path, generation: int, working: dict[str, Any], events: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    paths = generation_paths(run_root, generation)
    state = generation_state(working)
    effective, mutation = generation_training_config(config, generation)
    atomic_json(paths["root"] / "policy_mutation.json", mutation)
    rollouts = ensure_rollouts(config=effective, state=state, generation=generation, paths=paths)
    checkpoint = ensure_candidate_checkpoint(
        config=effective, state=state, generation=generation, paths=paths, rollouts=rollouts, log=events,
    )
    checkpoint = ensure_search_distilled_checkpoint(
        config=effective, generation=generation, paths=paths, actor_checkpoint=checkpoint, events=events,
    )
    q_checkpoint = ensure_action_q_checkpoint(
        config=effective, state=state, generation=generation, paths=paths,
        actor_checkpoint=checkpoint, events=events,
    )
    package = ensure_candidate_package(
        config=effective, state=state, generation=generation, paths=paths,
        checkpoint=checkpoint, q_checkpoint=q_checkpoint,
    )
    return package, {"pbt_mutation": mutation, "rollout_shards": len(rollouts)}


def produce_deck_candidate(
    config: dict[str, Any], async_config: dict[str, Any], run_root: Path,
    generation: int, working: dict[str, Any], events: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    paths = generation_paths(run_root, generation)
    parent = Path(working["training_parent_package"])
    mutation_root = paths["root"] / "deck_mutation"
    mutation_manifest = mutation_root / "manifest.json"
    if not mutation_manifest.is_file():
        run_process(
            host=config["local_host"], local_host=config["local_host"],
            command=[
                config["python"], str(Path(config["code_root"]) / "scripts" / "mutate_legal_decks.py"),
                "--base-deck", str(parent / "deck.csv"),
                "--cards", config["card_database"],
                "--pool-manifest", config["public_opponent_pool"],
                "--count", "1",
                "--max-swaps", str(async_config.get("deck_max_swaps", 3)),
                "--seed", str(int(config["base_seed"]) + generation * 1_000_000 + 701),
                "--output", str(mutation_root),
            ],
            log_path=mutation_root / "mutate.log",
        )
    mutation = read_json(mutation_manifest)["candidates"][0]
    package = paths["candidate"]
    if not (package / "agent_manifest.json").is_file():
        command = [
            config["python"], str(Path(config["code_root"]) / "scripts" / "materialize_rl_specialist_agent.py"),
            "--checkpoint", str(parent / "checkpoint.pt"),
            "--deck", str(mutation["deck"]),
            "--output", str(package),
            "--name", f"deck_candidate_g{generation:05d}",
        ]
        q_checkpoint = parent / "action_q.pt"
        if q_checkpoint.is_file() and bool(config.get("attach_action_q", True)):
            command.extend(["--q-checkpoint", str(q_checkpoint)])
            command.extend(q_materialization_arguments(config))
        run_process(
            host=config["local_host"], local_host=config["local_host"], command=command,
            log_path=paths["root"] / "materialize.log",
        )
    events.append({"event": "deck_mutation", "generation": generation, "mutation": mutation})
    return package, {"deck_mutation": mutation}


def find_duplicate(run_root: Path, candidate_sha256: str, generation: int) -> str | None:
    champion = read_json(run_root / "champion.json")
    if champion.get("candidate_sha256") == candidate_sha256:
        return "champion"
    for root in candidate_roots(run_root):
        manifest_path = root / "candidate.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        if int(manifest["generation"]) == generation:
            continue
        if manifest.get("candidate_sha256") == candidate_sha256:
            return str(root)
    return None


def produce_one(config: dict[str, Any], async_config: dict[str, Any], run_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    generation = int(state.get("active_generation") or state["next_generation"])
    paths = generation_paths(run_root, generation)
    paths["root"].mkdir(parents=True, exist_ok=True)
    working = production_snapshot(run_root, generation, async_config)
    state.update({"status": "producing", "active_generation": generation, "updated_at": now()})
    atomic_json(run_root / "producer_state.json", state)
    events: list[dict[str, Any]] = []
    if working["kind"] == "deck":
        package, method = produce_deck_candidate(config, async_config, run_root, generation, working, events)
        priority = int(async_config.get("deck_priority", 90))
    else:
        package, method = produce_policy_candidate(config, run_root, generation, working, events)
        priority = int(async_config.get("policy_priority", 100))
    fingerprint = package_fingerprint(package)
    manifest = {
        "schema_version": 1,
        "generation": generation,
        "kind": working["kind"],
        "package": str(package),
        "training_parent_version": working["training_parent_version"],
        "training_parent_package": working["training_parent_package"],
        "training_parent_checkpoint_sha256": working["training_parent_checkpoint_sha256"],
        "priority": priority,
        "queued_at": now(),
        "method": method,
        **fingerprint,
    }
    duplicate_of = find_duplicate(run_root, fingerprint["candidate_sha256"], generation)
    atomic_json(paths["root"] / "candidate.json", manifest)
    atomic_json(paths["root"] / "lifecycle.json", {
        "schema_version": 1,
        "status": "duplicate" if duplicate_of else "queued",
        "duplicate_of": duplicate_of,
        "updated_at": now(),
    })
    for event in events:
        append_event(run_root / "producer_events.jsonl", event)
    append_event(run_root / "producer_events.jsonl", {
        "event": "candidate_queued" if not duplicate_of else "candidate_duplicate",
        "generation": generation,
        "kind": working["kind"],
        "candidate_sha256": fingerprint["candidate_sha256"],
        "duplicate_of": duplicate_of,
    })
    state.update({
        "status": "queued" if not duplicate_of else "duplicate",
        "active_generation": None,
        "next_generation": generation + 1,
        "produced": int(state.get("produced", 0)) + 1,
        "last_candidate": str(paths["root"]),
        "updated_at": now(),
    })
    atomic_json(run_root / "producer_state.json", state)
    return state


def update_league(config: dict[str, Any], run_root: Path, package: str) -> None:
    path = run_root / "league.json"
    payload = read_json(path)
    packages = [value for value in payload["packages"] if value != package]
    packages.append(package)
    payload.update({
        "packages": packages[-int(config["population_limit"]):],
        "updated_at": now(),
    })
    atomic_json(path, payload)


def promote_compare_and_swap(
    run_root: Path, manifest: dict[str, Any], report: dict[str, Any], expected_version: int,
) -> bool:
    champion_path = run_root / "champion.json"
    champion = read_json(champion_path)
    if int(champion["version"]) != int(expected_version):
        return False
    new_version = expected_version + 1
    atomic_json(champion_path, {
        "schema_version": 1,
        "version": new_version,
        "package": manifest["package"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "deck_sha256": manifest["deck_sha256"],
        "action_q_sha256": manifest.get("action_q_sha256"),
        "candidate_sha256": manifest["candidate_sha256"],
        "source": f"generation_{int(manifest['generation']):05d}",
        "previous_version": expected_version,
        "previous_package": champion["package"],
        "promotion_report": report.get("promotion_report"),
        "updated_at": now(),
    })
    append_event(run_root / "champion_history.jsonl", {
        "event": "champion_promoted",
        "version": new_version,
        "generation": manifest["generation"],
        "candidate_sha256": manifest["candidate_sha256"],
        "previous_package": champion["package"],
        "package": manifest["package"],
    })
    return True


def evaluate_one(
    config: dict[str, Any], async_config: dict[str, Any], run_root: Path,
    root: Path, manifest: dict[str, Any], life: dict[str, Any],
) -> None:
    champion = read_json(run_root / "champion.json")
    champion_version = int(champion["version"])
    lag = champion_version - int(manifest["training_parent_version"])
    if lag > int(async_config.get("max_parent_version_lag", 2)):
        atomic_json(root / "lifecycle.json", {
            **life, "status": "stale_discarded", "parent_version_lag": lag, "updated_at": now(),
        })
        append_event(run_root / "evaluator_events.jsonl", {
            "event": "candidate_stale_discarded", "generation": manifest["generation"], "lag": lag,
        })
        return
    gate_root = root / f"gate_parent_v{champion_version:05d}"
    eval_paths = generation_paths(run_root, int(manifest["generation"]))
    eval_paths["gate"] = gate_root
    evaluating = {
        **life,
        "status": "evaluating",
        "evaluation_parent_version": champion_version,
        "evaluation_parent_package": champion["package"],
        "evaluation_started_at": life.get("evaluation_started_at") or now(),
        "updated_at": now(),
    }
    atomic_json(root / "lifecycle.json", evaluating)
    report = ensure_staged_gate(
        config=config,
        state={"champion_package": champion["package"]},
        generation=int(manifest["generation"]),
        paths=eval_paths,
        candidate_package=Path(manifest["package"]),
        public_items=load_manifest(Path(config["public_opponent_pool"])),
    )
    report_path = gate_root / "promotion.json"
    report_ref = {**report, "promotion_report": str(report_path)}
    current = read_json(run_root / "champion.json")
    if int(current["version"]) != champion_version:
        atomic_json(root / "lifecycle.json", {
            **evaluating,
            "status": "queued",
            "reason": "champion_changed_during_evaluation",
            "completed_gate": str(report_path),
            "updated_at": now(),
        })
        return
    promoted = bool(report["promote"]) and promote_compare_and_swap(
        run_root, manifest, report_ref, champion_version,
    )
    retained = False
    if promoted:
        update_league(config, run_root, manifest["package"])
    elif retain_candidate_in_league(config, report):
        update_league(config, run_root, manifest["package"])
        retained = True
    status = "promoted" if promoted else "rejected"
    atomic_json(root / "lifecycle.json", {
        **evaluating,
        "status": status,
        "promoted": promoted,
        "league_retained": retained,
        "promotion_report": str(report_path),
        "failed_stage": report.get("failed_stage"),
        "updated_at": now(),
    })
    append_event(run_root / "evaluator_events.jsonl", {
        "event": f"candidate_{status}",
        "generation": manifest["generation"],
        "kind": manifest["kind"],
        "evaluation_parent_version": champion_version,
        "failed_stage": report.get("failed_stage"),
        "league_retained": retained,
    })


def run_producer(config: dict[str, Any], async_config: dict[str, Any], run_root: Path) -> None:
    lock_handle = acquire_role_lock(run_root, "producer")
    state = producer_state(run_root)
    atomic_json(run_root / "producer_state.json", state)
    failures = 0
    try:
        while not (run_root / "STOP").exists():
            if count_pending(run_root) >= int(async_config["max_pending_candidates"]):
                state.update({"status": "waiting_for_queue_capacity", "updated_at": now()})
                atomic_json(run_root / "producer_state.json", state)
                time.sleep(float(async_config.get("producer_poll_seconds", 15)))
                continue
            try:
                state = produce_one(config, async_config, run_root, state)
                failures = 0
            except Exception as exc:
                failures += 1
                state.update({
                    "status": "error",
                    "consecutive_failures": failures,
                    "last_error": f"{type(exc).__name__}: {exc}",
                    "last_traceback": traceback.format_exc(),
                    "updated_at": now(),
                })
                atomic_json(run_root / "producer_state.json", state)
                if failures >= int(async_config.get("max_consecutive_failures", 3)):
                    state["status"] = "blocked"
                    atomic_json(run_root / "producer_state.json", state)
                    break
                time.sleep(float(async_config.get("failure_backoff_seconds", 60)))
            else:
                time.sleep(float(async_config.get("producer_sleep_seconds", 5)))
    finally:
        lock_handle.close()


def run_evaluator(config: dict[str, Any], async_config: dict[str, Any], run_root: Path) -> None:
    lock_handle = acquire_role_lock(run_root, "evaluator")
    state_path = run_root / "evaluator_state.json"
    state = read_json(state_path) if state_path.is_file() else {
        "schema_version": 1, "status": "initialized", "evaluated": 0, "updated_at": now(),
    }
    atomic_json(state_path, state)
    failures = 0
    try:
        while not (run_root / "STOP").exists():
            champion = read_json(run_root / "champion.json")
            selected = select_candidate(run_root, int(champion["version"]))
            if selected is None:
                state.update({"status": "waiting_for_candidate", "updated_at": now()})
                atomic_json(state_path, state)
                time.sleep(float(async_config.get("evaluator_poll_seconds", 10)))
                continue
            root, manifest, life = selected
            state.update({
                "status": "evaluating",
                "active_generation": manifest["generation"],
                "active_candidate": str(root),
                "updated_at": now(),
            })
            atomic_json(state_path, state)
            try:
                evaluate_one(config, async_config, run_root, root, manifest, life)
                failures = 0
                state.update({
                    "status": "completed_candidate",
                    "active_generation": None,
                    "active_candidate": None,
                    "evaluated": int(state.get("evaluated", 0)) + 1,
                    "updated_at": now(),
                })
                atomic_json(state_path, state)
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
                if failures >= int(async_config.get("max_consecutive_failures", 3)):
                    state["status"] = "blocked"
                    atomic_json(state_path, state)
                    break
                time.sleep(float(async_config.get("failure_backoff_seconds", 60)))
    finally:
        lock_handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Asynchronous Gold V3.1 candidate producer and evaluator")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--role", choices=("producer", "evaluator"), required=True)
    args = parser.parse_args()
    if os.name == "nt":
        raise RuntimeError("async Gold V3.1 services must run on a Linux training server")
    config, async_config = load_config(args.config)
    run_root = Path(config["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    initialize_registries(config, run_root)
    append_event(run_root / f"{args.role}_events.jsonl", {
        "event": "service_started", "role": args.role, "host": socket.gethostname(), "pid": os.getpid(),
    })
    if args.role == "producer":
        run_producer(config, async_config, run_root)
    else:
        run_evaluator(config, async_config, run_root)


if __name__ == "__main__":
    main()
