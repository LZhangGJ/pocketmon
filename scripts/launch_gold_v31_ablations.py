from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.continuous_rl_pipeline import ensure_staged_gate


PYTHON = Path("/homes/lzhang/mypath/new/envs/trans/bin/python")
SHARED = Path("/homes/lzhang/pocketmon")
RESULT_ROOT = SHARED / "results" / "gold_v31" / "ablations"
DECK_MAP = SHARED / "data" / "processed" / "replay_decks_2026-08-05_2026-08-06.jsonl.gz"
CARD_DATABASE = SHARED / "data" / "reference" / "official_cards.json"
ATTACK_DATABASE = SHARED / "data" / "reference" / "official_attacks.json"
CG_DIR = SHARED / "tmp" / "benchmark_agents" / "grim_hybrid_909" / "cg"
OFFICIAL_RANDOM = SHARED / "agents" / "official_random"
ARCHITECTURE = "structured_temporal_resource_belief_transformer_masked_pointer_with_stop"
HISTORY_TOKEN = "prior pre-action state plus that prior selected-option summary"


SPECIALISTS = {
    "garchomp": {
        "replay": SHARED / "results" / "gold_v3" / "bootstrap" / "garchomp" / "replay.jsonl.gz",
        "deck": SHARED / "results" / "gold_v3" / "bootstrap" / "garchomp" / "agent" / "deck.csv",
        "parent": SHARED / "results" / "gold_v3" / "bootstrap" / "garchomp" / "agent",
        "pool": ROOT / "configs" / "opponent_pool_gold_v3_garchomp.json",
        "seeds": [131, 132],
    },
    "grimmsnarl": {
        "replay": SHARED / "results" / "gold_v3" / "bootstrap" / "grimmsnarl" / "replay.jsonl.gz",
        "deck": SHARED / "results" / "gold_v3" / "bootstrap" / "grimmsnarl" / "agent" / "deck.csv",
        "parent": SHARED / "results" / "gold_v3" / "bootstrap" / "grimmsnarl" / "agent",
        "pool": ROOT / "configs" / "opponent_pool_gold_v3_grimmsnarl.json",
        "seeds": [137, 138],
    },
}

ARMS = {
    "history": {"history_length": 32, "resources": False, "belief": False},
    "resources": {"history_length": 0, "resources": True, "belief": False},
    "belief": {"history_length": 0, "resources": False, "belief": True},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
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
        raise RuntimeError("formal Gold V3.1 ablations require a clean worktree")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def freeze_prototypes(name: str, spec: dict[str, Any]) -> Path:
    output = RESULT_ROOT / name / "opponent_prototypes.json"
    if output.is_file():
        return output
    subprocess.run([
        str(PYTHON), str(ROOT / "scripts" / "freeze_opponent_deck_prototypes.py"),
        "--pool", str(spec["pool"]), "--output", str(output),
    ], cwd=ROOT, check=True)
    return output


def planned_config(
    name: str,
    spec: dict[str, Any],
    arm_name: str,
    arm: dict[str, Any],
    seed: int,
    prototypes: Path,
) -> dict[str, Any]:
    use_belief = bool(arm["belief"])
    history_length = int(arm["history_length"])
    parent_checkpoint = Path(spec["parent"]) / "checkpoint.pt"
    prototype_payload = json.loads(prototypes.read_text(encoding="utf-8"))
    prototype_count = len(prototype_payload["prototypes"]) if use_belief else 0
    history = {
        "enabled": history_length > 0,
        "encoder": "causal_transformer" if history_length else "none",
        "max_length": history_length,
    }
    if history_length:
        history.update({
            "group_by": ["episode_id", "player"],
            "order_by": "action_step",
            "token": HISTORY_TOKEN,
        })
    return {
        "experiment_id": f"GOLD-V31-{name.upper()}-{arm_name.upper()}-SEED-{seed}",
        "arm": arm_name,
        "architecture": ARCHITECTURE,
        "input": str(spec["replay"]),
        "input_sha256": sha256(Path(spec["replay"])),
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
        "history": history,
        "structured": {
            "card_attack_embeddings": True,
            "entity_encoder": "transformer_cross_attention",
            "deck_conditioning": "acting_player_submitted_deck_masked_mean",
            "card_text_embedding": "signed_hash_word_bigram_char3_128d",
            "deck_map_sha256": sha256(DECK_MAP),
            "card_database_sha256": sha256(CARD_DATABASE),
            "attack_database_sha256": sha256(ATTACK_DATABASE),
            "confidence_threshold": 0.0,
            "temporal_history": history_length > 0,
            "remaining_card_context": bool(arm["resources"]),
            "opponent_deck_belief": use_belief,
            "hidden_opponent_cards_used": False,
            "opponent_prototype_count": prototype_count,
            "opponent_prototypes_sha256": sha256(prototypes) if use_belief else None,
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
        "initialization_checkpoint_sha256": sha256(parent_checkpoint),
        "offline_rl": False,
    }


def training_command(
    name: str,
    spec: dict[str, Any],
    arm_name: str,
    arm: dict[str, Any],
    seed: int,
    prototypes: Path,
    root: Path,
) -> list[str]:
    command = [
        str(PYTHON), str(ROOT / "scripts" / "train_rl_policy.py"),
        "--input", str(spec["replay"]),
        "--experiment-id", f"GOLD-V31-{name.upper()}-{arm_name.upper()}-SEED-{seed}",
        "--planned-config", str(root / "planned_config.json"),
        "--seed", str(seed), "--split-seed", "20260807", "--validation-fraction", "0.2",
        "--epochs", "2", "--batch-size", "384", "--learning-rate", "0.00002",
        "--hidden-dim", "192", "--patience", "2", "--value-loss-weight", "0.10",
        "--gradient-clip-norm", "1.0", "--architecture", ARCHITECTURE,
        "--history-length", str(arm["history_length"]), "--deck-map", str(DECK_MAP),
        "--card-database", str(CARD_DATABASE), "--attack-database", str(ATTACK_DATABASE),
        "--confidence-threshold", "0.0", "--recency-half-life-days", "2.0",
        "--deck-stratification-alpha", "0.5", "--rating-stratification-alpha", "0.5",
        "--rating-bin-width", "100.0", "--min-sample-weight", "0.25", "--max-sample-weight", "4.0",
        "--device", "cuda:0", "--checkpoint-dir", str(root / "checkpoints"),
        "--metrics-output", str(root / "metrics.json"), "--split-output", str(root / "split.json"),
        "--runs-output", str(root / "runs.csv"),
        "--initialize-from", str(Path(spec["parent"]) / "checkpoint.pt"),
    ]
    if arm["resources"]:
        command.append("--v31-use-resources")
    if arm["belief"]:
        command.extend(["--v31-use-opponent-belief", "--opponent-prototypes", str(prototypes)])
    return command


def run_training_jobs(jobs: list[tuple[str, list[str], Path]], gpus: list[int]) -> None:
    for offset in range(0, len(jobs), len(gpus)):
        processes = []
        for slot, (label, command, log_path) in enumerate(jobs[offset:offset + len(gpus)]):
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            remote = shlex.join(["env", f"CUDA_VISIBLE_DEVICES={gpus[slot]}", *command])
            process = subprocess.Popen(["ssh", "doraemon20", remote], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
            processes.append((label, process, handle))
        failures = []
        for label, process, handle in processes:
            code = process.wait()
            handle.close()
            if code:
                failures.append((label, code))
        if failures:
            raise RuntimeError(f"V3.1 training jobs failed: {failures}")


def materialize_and_smoke(name: str, spec: dict[str, Any], root: Path, seed: int) -> Path:
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("status") != "completed_formal" or metrics.get("missing_seeds"):
        raise RuntimeError(f"{name}/{root.name} did not complete formally")
    if int(metrics["validation"].get("invalid_actions", -1)) != 0:
        raise RuntimeError(f"{name}/{root.name} produced invalid validation actions")
    checkpoint = Path(metrics["checkpoint"]["path"])
    package = root / "agent"
    if not (package / "agent_manifest.json").is_file():
        subprocess.run([
            str(PYTHON), str(ROOT / "scripts" / "materialize_rl_specialist_agent.py"),
            "--checkpoint", str(checkpoint), "--deck", str(spec["deck"]),
            "--output", str(package), "--name", f"gold_v31_{name}_{root.parent.name}_{seed}",
        ], cwd=ROOT, check=True)
    smoke_path = root / "smoke.json"
    if not smoke_path.is_file():
        games = []
        for learner_seat in (0, 1):
            completed = subprocess.run([
                str(PYTHON), str(ROOT / "scripts" / "run_local_match.py"),
                "--agent0", str(package if learner_seat == 0 else OFFICIAL_RANDOM),
                "--agent1", str(OFFICIAL_RANDOM if learner_seat == 0 else package),
                "--cg-dir", str(CG_DIR), "--seed", str(20260831000 + seed * 2 + learner_seat),
                "--max-decisions", "5000",
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            result = json.loads(completed.stdout.strip().splitlines()[-1])
            diagnostics = result["agent_diagnostics"][learner_seat]
            for key in ("load_errors", "inference_errors", "illegal_model_actions", "illegal_fallback_actions"):
                if int(diagnostics.get(key, 0)):
                    raise RuntimeError(f"{name}/{root.name} smoke {key}={diagnostics.get(key)}")
            games.append(result)
        atomic_json(smoke_path, {"games": games, "completed_at": now()})
    return package


def gate_config(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "python": str(PYTHON), "code_root": str(ROOT), "local_host": "doraemon03",
        "evaluation_hosts": ["doraemon15"], "cg_dir": str(CG_DIR),
        "public_opponent_pool": str(spec["pool"]),
        "evaluation_lock_path": str(SHARED / "results" / "continuous_rl" / "global_evaluation.lock"),
        "host_environment": {"doraemon15": {"LD_PRELOAD": "/homes/lzhang/mypath/new/envs/wsl4mis/lib/libstdc++.so.6"}},
        "base_seed": 20260831000,
        "min_head_to_head_score": 0.55, "min_head_to_head_wilson": 0.45,
        "min_public_delta": 0.02, "max_worst_matchup_regression": 0.10, "max_seat_gap": 0.12,
        "gate_stages": [
            {"name": "smoke20", "target_games": 20, "opponent_count": 4, "games_per_public": 2, "parent_games": 4,
             "min_head_to_head_score": 0.25, "min_head_to_head_wilson": 0.0, "min_public_delta": -0.25,
             "max_worst_matchup_regression": 0.50, "max_seat_gap": 1.0},
            {"name": "screen200", "target_games": 200, "opponent_count": 8, "games_per_public": 8, "parent_games": 72,
             "min_head_to_head_score": 0.50, "min_head_to_head_wilson": 0.35, "min_public_delta": 0.0,
             "max_worst_matchup_regression": 0.20, "max_seat_gap": 0.20},
            {"name": "confirm400", "target_games": 400, "opponent_count": 16, "games_per_public": 10, "parent_games": 80,
             "min_head_to_head_score": 0.55, "min_head_to_head_wilson": 0.45, "min_public_delta": 0.02,
             "max_worst_matchup_regression": 0.10, "max_seat_gap": 0.12},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible Gold V3.1 one-factor Transformer ablations")
    parser.add_argument("--specialist", choices=("all", *SPECIALISTS), default="all")
    args = parser.parse_args()
    selected = SPECIALISTS if args.specialist == "all" else {args.specialist: SPECIALISTS[args.specialist]}
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    state_path = RESULT_ROOT / "state.json"
    state: dict[str, Any] = {"status": "preparing", "started_at": now(), "kaggle_submission": "disabled"}
    atomic_json(state_path, state)
    try:
        state["git_sha"] = require_clean_worktree()
        jobs = []
        roots: dict[tuple[str, str, int], Path] = {}
        prototype_paths = {}
        for name, spec in selected.items():
            prototype_paths[name] = freeze_prototypes(name, spec)
            for arm_name, arm in ARMS.items():
                for seed in spec["seeds"]:
                    root = RESULT_ROOT / name / arm_name / f"seed_{seed}"
                    root.mkdir(parents=True, exist_ok=True)
                    roots[(name, arm_name, seed)] = root
                    plan = planned_config(name, spec, arm_name, arm, seed, prototype_paths[name])
                    if not (root / "planned_config.json").is_file():
                        atomic_json(root / "planned_config.json", plan)
                    if not (root / "metrics.json").is_file():
                        jobs.append((
                            f"{name}:{arm_name}:{seed}",
                            training_command(name, spec, arm_name, arm, seed, prototype_paths[name], root),
                            root / "train.log",
                        ))
        state.update({"status": "training", "jobs": len(jobs), "updated_at": now()})
        atomic_json(state_path, state)
        run_training_jobs(jobs, [1, 4, 0, 6])

        reports: dict[str, Any] = {}
        for name, spec in selected.items():
            reports[name] = {}
            for arm_name in ARMS:
                seed_reports = []
                for seed in spec["seeds"]:
                    root = roots[(name, arm_name, seed)]
                    package = materialize_and_smoke(name, spec, root, seed)
                    state.update({
                        "status": "gating", "current": f"{name}:{arm_name}:{seed}", "updated_at": now()
                    })
                    atomic_json(state_path, state)
                    report = ensure_staged_gate(
                        config=gate_config(spec),
                        state={"champion_package": str(spec["parent"])},
                        generation=seed,
                        paths={"gate": root / "gate"},
                        candidate_package=package,
                        public_items=json.loads(Path(spec["pool"]).read_text(encoding="utf-8")),
                    )
                    seed_reports.append({"seed": seed, "package": str(package), "report": report})
                reproducible = all(item["report"]["promote"] for item in seed_reports)
                reports[name][arm_name] = {
                    "reproducible_gain": reproducible,
                    "independent_training_seeds": seed_reports,
                }
                atomic_json(RESULT_ROOT / name / arm_name / "summary.json", reports[name][arm_name])
        status = "complete_with_reproducible_gain" if any(
            arm["reproducible_gain"] for specialist in reports.values() for arm in specialist.values()
        ) else "complete_no_reproducible_gain"
        state.update({"status": status, "reports": reports, "completed_at": now()})
        atomic_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False))
    except Exception as exc:
        state.update({"status": "error", "error": f"{type(exc).__name__}: {exc}", "updated_at": now()})
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
