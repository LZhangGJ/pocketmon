from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/homes/lzhang/mypath/new/envs/trans/bin/python")
SHARED = Path("/homes/lzhang/pocketmon")
BOOTSTRAP_ROOT = SHARED / "results" / "gold_v3" / "bootstrap"
REPLAY = SHARED / "data" / "processed" / "public_replay_2026-08-05_2026-08-06.jsonl.gz"
DECK_MAP = SHARED / "data" / "processed" / "replay_decks_2026-08-05_2026-08-06.jsonl.gz"
CARD_DATABASE = SHARED / "data" / "reference" / "official_cards.json"
ATTACK_DATABASE = SHARED / "data" / "reference" / "official_attacks.json"
INITIAL_CHECKPOINT = SHARED / "results" / "our_agents_2026-08-07" / "transformer_text_g0" / "checkpoint.pt"
CG_DIR = SHARED / "tmp" / "benchmark_agents" / "grim_hybrid_909" / "cg"
OFFICIAL_RANDOM = SHARED / "agents" / "official_random"
OLD_ROOTS = (
    SHARED / "results" / "continuous_rl" / "lopunny_gold_v2_frozen_league",
    SHARED / "results" / "continuous_rl" / "transformer_text_gold_v2",
)


SPECIALISTS = {
    "garchomp": {
        "role": "specialist",
        "deck": SHARED / "tmp" / "frontier_agents_2026-08-07" / "garchomp_v28" / "deck.csv",
        "min_similarity": 0.75,
        "max_episode_players": 0,
        "seed": 31,
        "host": "doraemon20",
        "config": ROOT / "configs" / "continuous_rl_gold_v3_garchomp_specialist.json",
    },
    "grimmsnarl": {
        "role": "anchor",
        "deck": SHARED / "tmp" / "benchmark_agents" / "grim_hybrid_909" / "deck.csv",
        "min_similarity": 0.85,
        "max_episode_players": 1500,
        "seed": 37,
        "host": "doraemon03",
        "config": ROOT / "configs" / "continuous_rl_gold_v3_grimmsnarl_anchor.json",
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_clean_worktree() -> str:
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise RuntimeError("Gold V3 formal bootstrap requires a clean training worktree")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def require_v2_stopped() -> None:
    for root in OLD_ROOTS:
        if not (root / "STOP").is_file():
            raise RuntimeError(f"old pipeline has no STOP marker: {root}")
        if (root / "state.json").is_file():
            state = json.loads((root / "state.json").read_text(encoding="utf-8"))
            if state.get("active_generation") is not None or state.get("status") != "stopped":
                raise RuntimeError(f"old pipeline has not reached a safe boundary: {root}: {state.get('status')}")
    processes = subprocess.run(
        ["pgrep", "-af", "continuous_rl_pipeline.py|run_league_schedule.py|run_local_match.py|collect_ppo_rollouts.py|train_action_q.py|train_rl_policy.py"],
        capture_output=True,
        text=True,
    ).stdout
    stale = [line for line in processes.splitlines() if "gold_v3" not in line and "pgrep -af" not in line]
    if stale:
        raise RuntimeError(f"old training workers remain:\n{chr(10).join(stale)}")


def copy_immutable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(destination) != sha256_file(source):
            raise RuntimeError(f"immutable bootstrap asset differs: {destination}")
        return
    shutil.copy2(source, destination)


def run_logged(command: list[str], log_path: Path, *, host: str) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    if host == "doraemon03":
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
    else:
        remote = shlex.join(["env", "CUDA_VISIBLE_DEVICES=0", *command])
        process = subprocess.Popen(["ssh", host, remote], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    process._gold_v3_log_handle = handle  # type: ignore[attr-defined]
    return process


def finish_logged(label: str, process: subprocess.Popen) -> None:
    return_code = process.wait()
    process._gold_v3_log_handle.close()  # type: ignore[attr-defined]
    if return_code:
        raise RuntimeError(f"{label} failed with exit code {return_code}")


def planned_bc_config(
    *, name: str, seed: int, input_path: Path, deck_map_path: Path, checkpoint: Path,
) -> dict[str, Any]:
    return {
        "experiment_id": f"GOLD-V3-{name.upper()}-DECK-BC",
        "arm": f"recent-replay-{name}-deck-specific-transformer-text",
        "architecture": "structured_card_attack_transformer_text_deck_masked_pointer_with_stop",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "split": {"kind": "episode_id", "seed": 20260807, "train_fraction": 0.8, "validation_fraction": 0.2},
        "training": {
            "formal_seeds": [seed],
            "epochs": 2,
            "batch_size": 384,
            "learning_rate": 0.00002,
            "hidden_dim": 192,
            "early_stopping_patience": 2,
            "value_loss_weight": 0.10,
            "gradient_clip_norm": 1.0,
        },
        "history": {"enabled": False, "encoder": "none", "max_length": 0},
        "structured": {
            "card_attack_embeddings": True,
            "entity_encoder": "transformer_cross_attention",
            "deck_conditioning": "acting_player_submitted_deck_masked_mean",
            "card_text_embedding": "signed_hash_word_bigram_char3_128d",
            "deck_map_sha256": sha256_file(deck_map_path),
            "card_database_sha256": sha256_file(CARD_DATABASE),
            "attack_database_sha256": sha256_file(ATTACK_DATABASE),
            "confidence_threshold": 0.0,
        },
        "sampling": {
            "recency_half_life_days": 2.0,
            "deck_stratification_alpha": 0.5,
            "rating_stratification_alpha": 0.5,
            "rating_bin_width": 100.0,
            "min_sample_weight": 0.25,
            "max_sample_weight": 4.0,
            "opponent_identity": "submitted_deck_sha256_proxy",
            "agent_id_available": False,
        },
        "initialization": "warm_start",
        "random_initialization": False,
        "initialization_checkpoint_sha256": sha256_file(checkpoint),
        "policy_loss_rows": "winner_only_from_target_deck_family",
        "value_loss_rows": "both_outcomes_from_target_deck_family",
        "offline_rl": False,
    }


def filter_dataset(name: str, spec: dict[str, Any], root: Path) -> tuple[Path, Path]:
    target_deck = root / "deck.csv"
    copy_immutable(Path(spec["deck"]), target_deck)
    output = root / "replay.jsonl.gz"
    audit = root / "replay_audit.json"
    if output.is_file() and audit.is_file():
        return output, target_deck
    subprocess.run([
        str(PYTHON), str(ROOT / "scripts" / "filter_specialist_replays.py"),
        "--input", str(REPLAY),
        "--deck-map", str(DECK_MAP),
        "--target-deck", str(target_deck),
        "--output", str(output),
        "--audit-output", str(audit),
        "--min-similarity", str(spec["min_similarity"]),
        "--min-episode-players", "20",
        "--max-episode-players", str(spec.get("max_episode_players", 0)),
        "--selection-seed", str(20260807 + int(spec["seed"])),
    ], cwd=ROOT, check=True)
    return output, target_deck


def training_command(name: str, spec: dict[str, Any], root: Path, input_path: Path, plan: Path) -> list[str]:
    seed = int(spec["seed"])
    return [
        str(PYTHON), str(ROOT / "scripts" / "train_rl_policy.py"),
        "--input", str(input_path),
        "--experiment-id", f"GOLD-V3-{name.upper()}-DECK-BC",
        "--planned-config", str(plan),
        "--seed", str(seed),
        "--split-seed", "20260807",
        "--validation-fraction", "0.2",
        "--epochs", "2",
        "--batch-size", "384",
        "--learning-rate", "0.00002",
        "--hidden-dim", "192",
        "--patience", "2",
        "--value-loss-weight", "0.10",
        "--gradient-clip-norm", "1.0",
        "--architecture", "structured_card_attack_transformer_text_deck_masked_pointer_with_stop",
        "--history-length", "0",
        "--deck-map", str(DECK_MAP),
        "--card-database", str(CARD_DATABASE),
        "--attack-database", str(ATTACK_DATABASE),
        "--confidence-threshold", "0.0",
        "--recency-half-life-days", "2.0",
        "--deck-stratification-alpha", "0.5",
        "--rating-stratification-alpha", "0.5",
        "--rating-bin-width", "100.0",
        "--min-sample-weight", "0.25",
        "--max-sample-weight", "4.0",
        "--device", "cuda:0",
        "--checkpoint-dir", str(root / "checkpoints"),
        "--metrics-output", str(root / "metrics.json"),
        "--split-output", str(root / "split.json"),
        "--runs-output", str(root / "runs.csv"),
        "--initialize-from", str(INITIAL_CHECKPOINT),
    ]


def materialize_and_smoke(name: str, root: Path, deck: Path) -> dict[str, Any]:
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("status") != "completed_formal" or metrics.get("missing_seeds"):
        raise RuntimeError(f"{name} BC did not complete formally")
    if int((metrics.get("validation") or {}).get("invalid_actions", -1)) != 0:
        raise RuntimeError(f"{name} BC has invalid validation actions")
    checkpoint = Path(metrics["checkpoint"]["path"])
    package = root / "agent"
    if not (package / "agent_manifest.json").is_file():
        subprocess.run([
            str(PYTHON), str(ROOT / "scripts" / "materialize_rl_specialist_agent.py"),
            "--checkpoint", str(checkpoint),
            "--deck", str(deck),
            "--output", str(package),
            "--name", f"gold_v3_{name}_deck_bc_g0",
        ], cwd=ROOT, check=True)
    smoke = []
    for learner_seat in (0, 1):
        command = [
            str(PYTHON), str(ROOT / "scripts" / "run_local_match.py"),
            "--agent0", str(package if learner_seat == 0 else OFFICIAL_RANDOM),
            "--agent1", str(OFFICIAL_RANDOM if learner_seat == 0 else package),
            "--cg-dir", str(CG_DIR),
            "--seed", str(20260807900 + learner_seat),
            "--max-decisions", "5000",
        ]
        completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        diagnostics = result["agent_diagnostics"][learner_seat]
        for key in ("load_errors", "inference_errors", "illegal_model_actions", "illegal_fallback_actions"):
            if int(diagnostics.get(key, 0)) != 0:
                raise RuntimeError(f"{name} smoke diagnostic {key}={diagnostics.get(key)}")
        smoke.append(result)
    atomic_json(root / "smoke.json", {"games": smoke, "completed_at": now()})
    return {"package": str(package), "checkpoint": metrics["checkpoint"], "smoke_games": len(smoke)}


def start_coordinator(config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_root = Path(config["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    if (run_root / "STOP").exists():
        raise RuntimeError(f"Gold V3 STOP marker exists: {run_root}")
    log_handle = (run_root / "coordinator.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        [str(PYTHON), str(ROOT / "scripts" / "continuous_rl_pipeline.py"), "--config", str(config_path)],
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (run_root / "coordinator.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    return {"config": str(config_path), "run_root": str(run_root), "pid": process.pid}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and launch deck-specialized Gold V3 training")
    parser.add_argument("--skip-v2-check", action="store_true", help="Only for isolated tests")
    args = parser.parse_args()
    BOOTSTRAP_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = BOOTSTRAP_ROOT / "launch_state.json"
    state: dict[str, Any] = {"status": "validating", "started_at": now()}
    atomic_json(state_path, state)
    try:
        commit = require_clean_worktree()
        if not args.skip_v2_check:
            require_v2_stopped()
        state.update({"status": "filtering_replays", "git_sha": commit, "updated_at": now()})
        atomic_json(state_path, state)

        prepared: dict[str, tuple[Path, Path]] = {}
        for name, spec in SPECIALISTS.items():
            root = BOOTSTRAP_ROOT / name
            root.mkdir(parents=True, exist_ok=True)
            input_path, deck = filter_dataset(name, spec, root)
            plan = planned_bc_config(
                name=name,
                seed=int(spec["seed"]),
                input_path=input_path,
                deck_map_path=DECK_MAP,
                checkpoint=INITIAL_CHECKPOINT,
            )
            atomic_json(root / "planned_config.json", plan)
            prepared[name] = (input_path, deck)

        state.update({"status": "training_deck_bc", "updated_at": now()})
        atomic_json(state_path, state)
        processes: dict[str, subprocess.Popen] = {}
        for name, spec in SPECIALISTS.items():
            root = BOOTSTRAP_ROOT / name
            if (root / "metrics.json").is_file():
                continue
            input_path, _ = prepared[name]
            command = training_command(name, spec, root, input_path, root / "planned_config.json")
            processes[name] = run_logged(command, root / "train.log", host=str(spec["host"]))
        for name, process in processes.items():
            finish_logged(name, process)

        state.update({"status": "materializing_and_smoke_testing", "updated_at": now()})
        atomic_json(state_path, state)
        bootstrap = {
            name: materialize_and_smoke(name, BOOTSTRAP_ROOT / name, prepared[name][1])
            for name in SPECIALISTS
        }
        state.update({"status": "starting_coordinators", "bootstrap": bootstrap, "updated_at": now()})
        atomic_json(state_path, state)
        coordinators = [start_coordinator(Path(spec["config"])) for spec in SPECIALISTS.values()]
        state.update({
            "status": "running",
            "coordinators": coordinators,
            "completed_at": now(),
            "division": {
                "doraemon03": "Grimmsnarl deck-BC, rollouts, PPO and coordinator",
                "doraemon20": "Garchomp deck-BC, counterfactual search, Q and rollouts",
                "doraemon15": "isolated serialized promotion evaluation only",
            },
        })
        atomic_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False))
    except Exception as exc:
        state.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "updated_at": now()})
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
