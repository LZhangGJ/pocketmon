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


def load_rollouts(
    paths: list[Path], behavior_sha256: str, teacher_sha256: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audits = []
    seen: set[tuple[str, int, int]] = set()
    for path in paths:
        count = 0
        episodes: set[str] = set()
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("rollout_format") != ROLLOUT_FORMAT:
                    raise Experiment7Error(f"unsupported rollout format in {path}")
                if row.get("behavior_checkpoint_sha256") != behavior_sha256:
                    raise Experiment7Error("rollout behavior checkpoint mismatch")
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
        audits.append({"path": str(path.resolve()), "sha256": sha256_file(path), "decisions": count, "episodes": len(episodes)})
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one Deck-8 Universal PPO generation")
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--rollouts", nargs="+", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--generation", type=int, required=True)
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
    parser.add_argument("--gradient-clip-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() or args.metrics_output.exists():
        raise FileExistsError("refusing to overwrite Universal PPO outputs")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise Experiment7Error("formal Universal PPO training requires a clean worktree")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, parent = load_universal_checkpoint(args.initialize_from, args.reference_root, device)
    behavior_sha = sha256_file(args.initialize_from)
    teacher_sha = sha256_file(args.teacher)
    rows, audits = load_rollouts(args.rollouts, behavior_sha, teacher_sha)
    rows = normalize_advantages(compute_gae(rows, args.gamma, args.gae_lambda))
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
        order = rng.permutation(len(rows))
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
        "parent": {"path": str(args.initialize_from.resolve()), "sha256": behavior_sha},
        "teacher": {"path": str(args.teacher.resolve()), "sha256": teacher_sha},
        "parentMetadata": parent.get("metadata", {}),
        "rollouts": audits,
        "epochs": epoch_rows,
        "stoppedForKl": stopped_for_kl,
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
