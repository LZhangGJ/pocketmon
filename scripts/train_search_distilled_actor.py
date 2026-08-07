from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.bc import TrajectoryDataset, make_loader
from rl.ppo import evaluate_action_sequences, load_checkpoint, sha256_file, to_device
from rl.reproducibility import seed_deterministically


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def assert_clean_worktree() -> None:
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise RuntimeError("formal search distillation requires a clean worktree")


def load_counterfactual_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audits = []
    seen = set()
    for path in paths:
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("rollout_format") != "counterfactual_action_q_v1":
                    raise ValueError(f"unsupported counterfactual row in {path}")
                key = (
                    str(row["episode_id"]), int(row["observation_step"]),
                    int(row["player"]), int(row["option_index"]),
                )
                if key in seen:
                    raise ValueError(f"duplicate counterfactual action: {key}")
                seen.add(key)
                rows.append(row)
                count += 1
        audits.append({"path": str(path), "sha256": sha256_file(path), "rows": count})
    return rows, audits


def build_teacher_rows(
    rows: list[dict[str, Any]], min_margin: float, max_target_std: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if min_margin < 0.0 or max_target_std < 0.0:
        raise ValueError("teacher margin and uncertainty thresholds must be non-negative")
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["episode_id"]), int(row["observation_step"]), int(row["player"]))].append(row)
    teachers = []
    counters = Counter(groups=len(groups))
    for group in groups.values():
        if len(group) < 2:
            counters["too_few_actions"] += 1
            continue
        ranked = sorted(group, key=lambda row: float(row["q_target"]), reverse=True)
        best, second = ranked[:2]
        margin = float(best["q_target"]) - float(second["q_target"])
        uncertainty = float(best.get("q_target_std", 0.0))
        if not math.isfinite(margin) or not math.isfinite(uncertainty):
            counters["non_finite"] += 1
            continue
        if margin < min_margin:
            counters["low_margin"] += 1
            continue
        if uncertainty > max_target_std:
            counters["high_uncertainty"] += 1
            continue
        teacher = dict(best)
        teacher["action"] = [int(best["option_index"])]
        teacher["teacher_margin"] = margin
        teacher["teacher_uncertainty"] = uncertainty
        teacher["teacher_weight"] = min(1.0, margin) / (1.0 + uncertainty)
        teachers.append(teacher)
        counters["accepted"] += 1
    return teachers, dict(counters)


def split_teacher_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes = sorted({str(row["episode_id"]) for row in rows})
    validation_ids = {episode for index, episode in enumerate(episodes) if index % 5 == 0}
    train = [row for row in rows if str(row["episode_id"]) not in validation_ids]
    validation = [row for row in rows if str(row["episode_id"]) in validation_ids]
    return (train or list(rows), validation or list(rows[: min(256, len(rows))]))


@torch.inference_mode()
def validate(model, rows: list[dict[str, Any]], batch_size: int, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = agreement = weight_sum = 0.0
    count = 0
    for original in make_loader(TrajectoryDataset(rows), batch_size, shuffle=False, seed=0):
        batch = to_device(original, device)
        log_probability, _, _ = evaluate_action_sequences(model, batch)
        state, encoded_options, _ = model.encode_batch(batch)
        selected = torch.zeros_like(batch["option_mask"])
        counts = torch.zeros(len(original["rows"]), dtype=torch.long, device=device)
        logits = model.pointer_logits(state, encoded_options, selected, counts)[:, :-1]
        logits = logits.masked_fill(~batch["option_mask"], torch.finfo(logits.dtype).min)
        targets = torch.tensor(
            [int(row["option_index"]) for row in original["rows"]], dtype=torch.long, device=device
        )
        weights = torch.tensor(
            [float(row["teacher_weight"]) for row in original["rows"]], device=device
        )
        total_loss += float((-log_probability * weights).sum())
        weight_sum += float(weights.sum())
        agreement += float((logits.argmax(dim=1) == targets).sum())
        count += len(original["rows"])
    return {
        "rows": count,
        "weighted_nll": total_loss / max(weight_sum, 1e-8),
        "top1_agreement": agreement / max(count, 1),
    }


def atomic_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill counterfactual search improvements into a PPO actor")
    parser.add_argument("--rows", nargs="+", required=True, type=Path)
    parser.add_argument("--initialize-from", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-6)
    parser.add_argument("--min-margin", type=float, default=0.15)
    parser.add_argument("--max-target-std", type=float, default=0.75)
    parser.add_argument("--anchor-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--gradient-clip-norm", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.metrics_output.exists():
        raise FileExistsError("refusing to overwrite search-distillation outputs")
    if min(args.epochs, args.batch_size) <= 0 or args.learning_rate <= 0.0:
        raise ValueError("invalid search-distillation training configuration")
    if not args.allow_dirty_smoke:
        assert_clean_worktree()
    seed_deterministically(args.seed)
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    ))
    started = time.perf_counter()
    model, parent = load_checkpoint(args.initialize_from, device)
    reference, _ = load_checkpoint(args.initialize_from, device)
    reference.eval()
    reference.requires_grad_(False)
    raw_rows, audits = load_counterfactual_rows(args.rows)
    teacher_rows, teacher_audit = build_teacher_rows(raw_rows, args.min_margin, args.max_target_std)
    if not teacher_rows:
        raise RuntimeError("counterfactual search produced no confident policy teachers")
    train_rows, validation_rows = split_teacher_rows(teacher_rows)
    before = validate(model, validation_rows, args.batch_size, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    epoch_metrics = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = Counter()
        for original in make_loader(
            TrajectoryDataset(train_rows), args.batch_size, shuffle=True, seed=args.seed + epoch
        ):
            batch = to_device(original, device)
            new_log_probability, entropy, _ = evaluate_action_sequences(model, batch)
            with torch.no_grad():
                reference_log_probability, _, _ = evaluate_action_sequences(reference, batch)
            weights = torch.tensor(
                [float(row["teacher_weight"]) for row in original["rows"]], device=device
            )
            denominator = weights.sum().clamp_min(1e-8)
            distillation_loss = -(new_log_probability * weights).sum() / denominator
            anchor_loss = (((new_log_probability - reference_log_probability) ** 2) * weights).sum() / denominator
            entropy_mean = (entropy * weights).sum() / denominator
            loss = (
                distillation_loss + args.anchor_coefficient * anchor_loss
                - args.entropy_coefficient * entropy_mean
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
            optimizer.step()
            totals["batches"] += 1
            totals["loss"] += float(loss.detach())
            totals["distillation_loss"] += float(distillation_loss.detach())
            totals["anchor_loss"] += float(anchor_loss.detach())
            totals["entropy"] += float(entropy_mean.detach())
            totals["gradient_norm"] += float(gradient_norm)
        batches = int(totals["batches"])
        record = {"epoch": epoch, "batches": batches, **{
            key: totals[key] / max(1, batches)
            for key in ("loss", "distillation_loss", "anchor_loss", "entropy", "gradient_norm")
        }}
        epoch_metrics.append(record)
        print(json.dumps(record), flush=True)
    after = validate(model, validation_rows, args.batch_size, device)
    distillation = {
        "method": "counterfactual_search_policy_distillation",
        "parent_checkpoint_sha256": sha256_file(args.initialize_from),
        "training_rows": audits,
        "teacher_audit": teacher_audit,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "min_margin": args.min_margin,
        "max_target_std": args.max_target_std,
        "anchor_coefficient": args.anchor_coefficient,
        "epochs": epoch_metrics,
        "validation_before": before,
        "validation_after": after,
    }
    payload = dict(parent)
    payload["model"] = model.state_dict()
    payload["optimizer"] = optimizer.state_dict()
    payload["search_distillation"] = distillation
    payload["code_commit"] = git_sha()
    payload["config"] = {
        **dict(parent.get("config") or {}),
        "training_method": "ppo_plus_counterfactual_search_distillation",
        "search_distillation_parent_sha256": distillation["parent_checkpoint_sha256"],
    }
    atomic_save(payload, args.output)
    metrics = {
        "status": "completed_formal" if not args.allow_dirty_smoke else "completed_smoke",
        **distillation,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "device": str(device),
        "runtime_seconds": time.perf_counter() - started,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
