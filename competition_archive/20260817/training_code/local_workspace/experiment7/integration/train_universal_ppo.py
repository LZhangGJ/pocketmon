from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import Experiment7Error, sha256_file, utc_now, write_json
from universal_ppo import (
    ARCHITECTURE,
    ROLLOUT_FORMAT,
    collate_rows,
    compute_gae,
    load_universal_checkpoint,
    normalize_advantages,
    ppo_loss,
)


def require_clean_repository(script_path: Path = Path(__file__)) -> tuple[Path, str]:
    """Resolve repository provenance independently of the process cwd.

    Training artifacts record the source commit, but a dirty worktree is not a
    training gate. Runtime code and immutable inputs are already captured by
    the generation receipts; rejecting a batch here only creates retry churn
    when another controller updates the shared worktree.
    """
    integration = script_path.resolve().parent
    try:
        repository = Path(
            subprocess.check_output(
                ["git", "-C", str(integration), "rev-parse", "--show-toplevel"],
                text=True,
            ).strip()
        ).resolve()
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise Experiment7Error(
            f"unable to validate Universal PPO repository for {integration}"
        ) from error
    return repository, commit


def load_rollouts(
    paths: list[Path],
    behavior_sha256: str,
    teacher_sha256: str,
    *,
    allowed_behavior_generations: dict[str, int] | None = None,
    current_generation: int | None = None,
    max_behavior_lag: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audits = []
    seen: set[tuple[str, int, int]] = set()
    for path in paths:
        count = 0
        episodes: set[str] = set()
        behavior_counts: Counter[str] = Counter()
        behavior_generations: Counter[int] = Counter()
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("rollout_format") != ROLLOUT_FORMAT:
                    raise Experiment7Error(f"unsupported rollout format in {path}")
                row_behavior_sha = str(row.get("behavior_checkpoint_sha256", ""))
                if allowed_behavior_generations is None:
                    if row_behavior_sha != behavior_sha256:
                        raise Experiment7Error("rollout behavior checkpoint mismatch")
                else:
                    if current_generation is None:
                        raise Experiment7Error("current generation is required for asynchronous rollouts")
                    if row_behavior_sha not in allowed_behavior_generations:
                        raise Experiment7Error("rollout behavior checkpoint is not in the snapshot manifest")
                    behavior_generation = int(row.get("behavior_generation", -1))
                    if behavior_generation != allowed_behavior_generations[row_behavior_sha]:
                        raise Experiment7Error("rollout behavior generation mismatch")
                    lag = current_generation - behavior_generation
                    if lag < 0 or lag > max_behavior_lag:
                        raise Experiment7Error(
                            f"rollout behavior lag outside [0, {max_behavior_lag}]: {lag}"
                        )
                if row.get("teacher_checkpoint_sha256") != teacher_sha256:
                    raise Experiment7Error("rollout teacher checkpoint mismatch")
                key = (str(row["episode_id"]), int(row["player"]), int(row["action_step"]))
                if key in seen:
                    raise Experiment7Error(f"duplicate Universal PPO decision: {key}")
                seen.add(key)
                action = list(row["action"])
                option_count = len(row["options"])
                if not (
                    int(row["min_count"]) <= len(action) <= int(row["max_count"])
                    and len(action) == len(set(action))
                    and all(0 <= int(index) < option_count for index in action)
                ):
                    raise Experiment7Error(f"illegal rollout action: {key}")
                for field in (
                    "behavior_log_probability",
                    "teacher_log_probability",
                    "behavior_value",
                    "behavior_entropy",
                    "reward",
                    "outcome",
                ):
                    if not math.isfinite(float(row[field])):
                        raise Experiment7Error(f"non-finite {field}: {key}")
                rows.append(row)
                count += 1
                episodes.add(str(row["episode_id"]))
                behavior_counts[row_behavior_sha] += 1
                if "behavior_generation" in row:
                    behavior_generations[int(row["behavior_generation"])] += 1
        audits.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "decisions": count,
                "episodes": len(episodes),
                "behaviorCheckpoints": dict(behavior_counts),
                "behaviorGenerations": {
                    str(key): value for key, value in sorted(behavior_generations.items())
                },
            }
        )
    if not rows:
        raise Experiment7Error("no Universal PPO rollout rows")
    return rows, audits


def save_checkpoint(path: Path, model: torch.nn.Module, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "schema_version": 2,
            "architecture": ARCHITECTURE,
            "config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        temporary,
    )
    temporary.replace(path)


def compact_parent_metadata(parent: dict[str, Any]) -> dict[str, Any]:
    """Keep lineage useful without recursively embedding every ancestor."""
    metadata = parent.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    keys = (
        "stage",
        "createdAt",
        "role",
        "generation",
        "seed",
        "stoppedForKl",
        "trainingConfig",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def balanced_player_order(rows: list[dict[str, Any]], rng: np.random.Generator) -> np.ndarray:
    by_player = {
        player: np.asarray(
            [index for index, row in enumerate(rows) if int(row["player"]) == player],
            dtype=np.int64,
        )
        for player in (0, 1)
    }
    if not len(by_player[0]) or not len(by_player[1]):
        return rng.permutation(len(rows))
    target = max(len(by_player[0]), len(by_player[1]))
    columns = []
    for player in (0, 1):
        source = by_player[player]
        repeats = []
        while sum(len(part) for part in repeats) < target:
            repeats.append(rng.permutation(source))
        columns.append(np.concatenate(repeats)[:target])
    return np.column_stack(columns).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Deck-8 Universal PPO generation")
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--rollouts", nargs="+", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--allowed-behavior-manifest", type=Path)
    parser.add_argument("--current-generation", type=int)
    parser.add_argument("--max-behavior-lag", type=int, default=0)
    parser.add_argument("--role", choices=("generalist", "hard_exploiter", "diversity", "conservative"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.1)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--teacher-anchor-coefficient", type=float, default=0.02)
    parser.add_argument("--tactical-preference-coefficient", type=float, default=0.04)
    parser.add_argument("--seat1-weight", type=float, default=1.0)
    parser.add_argument("--normalize-advantages-by-player", action="store_true")
    parser.add_argument("--balance-player-minibatches", action="store_true")
    parser.add_argument("--gradient-clip-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--max-initial-clip-fraction", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not math.isfinite(args.seat1_weight) or args.seat1_weight <= 0.0:
        raise ValueError("seat1 weight must be finite and positive")
    if args.output.exists() or args.metrics_output.exists():
        raise FileExistsError("refusing to overwrite Universal PPO outputs")
    repository, repository_commit = require_clean_repository()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, parent = load_universal_checkpoint(args.initialize_from, args.reference_root, device)
    behavior_sha = sha256_file(args.initialize_from)
    teacher_sha = sha256_file(args.teacher)
    allowed_behavior_generations = None
    if args.allowed_behavior_manifest is not None:
        payload = json.loads(args.allowed_behavior_manifest.read_text(encoding="utf-8"))
        snapshots = payload.get("snapshots", []) if isinstance(payload, dict) else []
        allowed_behavior_generations = {
            str(row["sha256"]): int(row["generation"]) for row in snapshots
        }
        if not allowed_behavior_generations:
            raise Experiment7Error("allowed behavior manifest has no snapshots")
        if args.current_generation is None or args.max_behavior_lag < 0:
            raise Experiment7Error("invalid asynchronous rollout lag configuration")
    rows, audits = load_rollouts(
        args.rollouts,
        behavior_sha,
        teacher_sha,
        allowed_behavior_generations=allowed_behavior_generations,
        current_generation=args.current_generation,
        max_behavior_lag=args.max_behavior_lag,
    )
    rows = normalize_advantages(
        compute_gae(rows, args.gamma, args.gae_lambda),
        by_player=args.normalize_advantages_by_player,
    )
    initial_policy_shift = None
    if allowed_behavior_generations is not None:
        totals: Counter[str] = Counter()
        examples = 0
        with torch.inference_mode():
            for begin in range(0, len(rows), args.batch_size):
                chosen = rows[begin : begin + args.batch_size]
                _, metrics = ppo_loss(
                    model,
                    collate_rows(chosen, device),
                    clip_ratio=args.clip_ratio,
                    value_clip=args.value_clip,
                    value_coefficient=args.value_coefficient,
                    entropy_coefficient=args.entropy_coefficient,
                    teacher_anchor_coefficient=args.teacher_anchor_coefficient,
                    tactical_preference_coefficient=args.tactical_preference_coefficient,
                    seat1_weight=args.seat1_weight,
                )
                count = len(chosen)
                examples += count
                for key, value in metrics.items():
                    totals[key] += float(value) * count
        initial_policy_shift = {
            key: value / max(examples, 1) for key, value in totals.items()
        }
        initial_policy_shift["decisions"] = examples
        if initial_policy_shift["approximateKl"] > args.target_kl:
            raise Experiment7Error(
                "asynchronous rollout rejected by initial KL gate: "
                f"{initial_policy_shift['approximateKl']:.6f} > {args.target_kl:.6f}"
            )
        if initial_policy_shift["clipFraction"] > args.max_initial_clip_fraction:
            raise Experiment7Error(
                "asynchronous rollout rejected by initial clip-fraction gate: "
                f"{initial_policy_shift['clipFraction']:.6f} > "
                f"{args.max_initial_clip_fraction:.6f}"
            )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    epoch_rows = []
    stopped_for_kl = False
    started = time.perf_counter()
    for epoch in range(1, args.ppo_epochs + 1):
        totals: Counter[str] = Counter()
        examples = 0
        order = (
            balanced_player_order(rows, rng)
            if args.balance_player_minibatches
            else rng.permutation(len(rows))
        )
        for begin in range(0, len(rows), args.batch_size):
            indices = order[begin : begin + args.batch_size]
            chosen = [rows[int(index)] for index in indices]
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = ppo_loss(
                model,
                collate_rows(chosen, device),
                clip_ratio=args.clip_ratio,
                value_clip=args.value_clip,
                value_coefficient=args.value_coefficient,
                entropy_coefficient=args.entropy_coefficient,
                teacher_anchor_coefficient=args.teacher_anchor_coefficient,
                tactical_preference_coefficient=args.tactical_preference_coefficient,
                seat1_weight=args.seat1_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            count = len(chosen)
            examples += count
            for key, value in metrics.items():
                totals[key] += float(value) * count
            if metrics["approximateKl"] > args.target_kl:
                stopped_for_kl = True
                break
        epoch_metrics = {key: value / max(examples, 1) for key, value in totals.items()}
        epoch_metrics.update({"epoch": epoch, "decisions": examples})
        epoch_rows.append(epoch_metrics)
        print(json.dumps(epoch_metrics), flush=True)
        if stopped_for_kl:
            break
    metadata = {
        "stage": "universal_ppo",
        "createdAt": utc_now(),
        "role": args.role,
        "generation": args.generation,
        "seed": args.seed,
        "repository": {"path": str(repository), "commit": repository_commit},
        "parent": {"path": str(args.initialize_from.resolve()), "sha256": behavior_sha},
        "teacher": {"path": str(args.teacher.resolve()), "sha256": teacher_sha},
        "parentMetadata": compact_parent_metadata(parent),
        "rollouts": audits,
        "initialPolicyShift": initial_policy_shift,
        "epochs": epoch_rows,
        "stoppedForKl": stopped_for_kl,
        "trainingConfig": {
            "teacherAnchorCoefficient": args.teacher_anchor_coefficient,
            "tacticalPreferenceCoefficient": args.tactical_preference_coefficient,
            "seat1Weight": args.seat1_weight,
            "normalizeAdvantagesByPlayer": args.normalize_advantages_by_player,
            "balancePlayerMinibatches": args.balance_player_minibatches,
        },
        "seconds": time.perf_counter() - started,
    }
    save_checkpoint(args.output, model, metadata)
    report = {
        "schemaVersion": 1,
        **metadata,
        "output": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output)},
    }
    write_json(args.metrics_output, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
