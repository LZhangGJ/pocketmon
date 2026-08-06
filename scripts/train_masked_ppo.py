from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.bc import TrajectoryDataset, action_is_legal, make_loader
from rl.ppo import (
    compute_gae,
    load_checkpoint,
    normalize_advantages,
    ppo_batch_loss,
    sha256_file,
    to_device,
)


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def assert_clean_worktree() -> None:
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise RuntimeError("formal PPO training requires a clean worktree")


def load_rollouts(paths: list[Path], expected_checkpoint_sha256: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audits = []
    seen_keys: set[tuple[str, int, int]] = set()
    for path in paths:
        path_rows = 0
        episodes: set[str] = set()
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema_version") != 3 or row.get("rollout_format") != "masked_ppo_v1":
                    raise ValueError(f"unsupported PPO row schema in {path}")
                if row.get("behavior_checkpoint_sha256") != expected_checkpoint_sha256:
                    raise ValueError("rollout behavior checkpoint does not match PPO parent")
                key = (str(row["episode_id"]), int(row["player"]), int(row["action_step"]))
                if key in seen_keys:
                    raise ValueError(f"duplicate PPO decision: {key}")
                seen_keys.add(key)
                if not action_is_legal(
                    list(row["action"]), len(row["options"]), int(row["min_count"]), int(row["max_count"])
                ):
                    raise ValueError(f"illegal PPO action in {key}")
                for field in ("behavior_log_probability", "behavior_value", "behavior_entropy", "reward", "outcome"):
                    if not math.isfinite(float(row[field])):
                        raise ValueError(f"non-finite {field} in {key}")
                rows.append(row)
                path_rows += 1
                episodes.add(str(row["episode_id"]))
        audits.append({
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": path_rows,
            "episodes": len(episodes),
        })
    if not rows:
        raise ValueError("no PPO rollout rows")
    return rows, audits


def fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def atomic_torch_save(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="On-policy masked autoregressive PPO with GAE")
    parser.add_argument("--rollouts", nargs="+", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.1)
    parser.add_argument("--value-clip", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.metrics_output.exists():
        raise FileExistsError("refusing to overwrite PPO output")
    if args.generation < 1 or args.ppo_epochs < 1 or args.batch_size < 1:
        raise ValueError("invalid PPO generation/epoch/batch configuration")
    if not args.allow_dirty_smoke:
        assert_clean_worktree()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    ))
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    parent_sha256 = sha256_file(args.initialize_from)
    model, parent = load_checkpoint(args.initialize_from, device)
    rows, rollout_audits = load_rollouts(args.rollouts, parent_sha256)
    rows = normalize_advantages(compute_gae(rows, gamma=args.gamma, gae_lambda=args.gae_lambda))
    advantages = [float(row["advantage"]) for row in rows]
    returns = [float(row["return"]) for row in rows]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    dataset = TrajectoryDataset(rows)
    epoch_metrics = []
    stopped_for_kl = False
    for epoch in range(1, args.ppo_epochs + 1):
        model.train()
        totals = Counter()
        loader = make_loader(dataset, args.batch_size, shuffle=True, seed=args.seed + epoch)
        for original in loader:
            batch = to_device(original, device)
            loss, parts = ppo_batch_loss(
                model,
                batch,
                clip_ratio=args.clip_ratio,
                value_clip=args.value_clip,
                value_coefficient=args.value_coefficient,
                entropy_coefficient=args.entropy_coefficient,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite PPO loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            totals["batches"] += 1
            totals["loss"] += float(loss.detach())
            totals["gradient_norm"] += float(gradient_norm)
            for key, value in parts.items():
                totals[key] += value
            if parts["approximate_kl"] > args.target_kl:
                stopped_for_kl = True
                break
        batches = int(totals["batches"])
        if not batches:
            raise RuntimeError("PPO processed no batches")
        record = {"epoch": epoch, **{
            key: totals[key] / batches
            for key in (
                "loss", "policy_loss", "value_loss", "entropy", "approximate_kl",
                "clip_fraction", "ratio_mean", "value_mean", "gradient_norm",
            )
        }, "batches": batches, "stopped_for_kl": stopped_for_kl}
        epoch_metrics.append(record)
        print(json.dumps(record), flush=True)
        if stopped_for_kl:
            break

    parent_config = dict(parent.get("config") or {})
    training_payload = {
        "method": "masked_autoregressive_ppo_gae",
        "generation": args.generation,
        "parent_checkpoint_sha256": parent_sha256,
        "rollouts": rollout_audits,
        "seed": args.seed,
        "ppo_epochs": args.ppo_epochs,
        "epochs_completed": len(epoch_metrics),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_ratio": args.clip_ratio,
        "value_clip": args.value_clip,
        "value_coefficient": args.value_coefficient,
        "entropy_coefficient": args.entropy_coefficient,
        "target_kl": args.target_kl,
    }
    policy_fingerprint = fingerprint({"code_commit": git_sha(), **training_payload})
    parent_config.update({
        "training_method": "masked_autoregressive_ppo_gae",
        "rl_generation": args.generation,
        "parent_checkpoint_sha256": parent_sha256,
        "policy_fingerprint": policy_fingerprint,
    })
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "hidden_dim": int(parent["hidden_dim"]),
        "config": parent_config,
        "ppo": training_payload,
        "code_commit": git_sha(),
    }
    atomic_torch_save(checkpoint, args.output)
    metrics = {
        "status": "completed_formal" if not args.allow_dirty_smoke else "completed_smoke",
        "generation": args.generation,
        "rows": len(rows),
        "episodes": len({str(row["episode_id"]) for row in rows}),
        "advantage_mean": sum(advantages) / len(advantages),
        "advantage_min": min(advantages),
        "advantage_max": max(advantages),
        "return_mean": sum(returns) / len(returns),
        "parent_checkpoint_sha256": parent_sha256,
        "candidate_checkpoint": str(args.output),
        "candidate_checkpoint_sha256": sha256_file(args.output),
        "policy_fingerprint": policy_fingerprint,
        "device": str(device),
        "runtime_seconds": time.perf_counter() - started,
        "epochs": epoch_metrics,
        "stopped_for_kl": stopped_for_kl,
        "rollouts": rollout_audits,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
