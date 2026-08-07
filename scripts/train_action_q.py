from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.action_q import ActionValueEnsemble
from rl.bc import TrajectoryDataset, make_loader
from rl.ppo import load_checkpoint, sha256_file, to_device


def assert_clean_worktree() -> None:
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if dirty.strip():
        raise RuntimeError("formal action-Q training requires a clean worktree")


def load_rows(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    audits = []
    seen = set()
    for path in paths:
        count = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema_version") != 1 or row.get("rollout_format") != "counterfactual_action_q_v1":
                    raise ValueError(f"unsupported action-Q row in {path}")
                key = (str(row["episode_id"]), int(row["observation_step"]), int(row["option_index"]))
                if key in seen:
                    raise ValueError(f"duplicate counterfactual row: {key}")
                seen.add(key)
                target = float(row["q_target"])
                if not math.isfinite(target) or not -1.0 <= target <= 1.0:
                    raise ValueError(f"invalid counterfactual target: {target}")
                if list(row["action"]) != [int(row["option_index"])]:
                    raise ValueError("action-Q row/action mismatch")
                rows.append(row)
                count += 1
        audits.append({"path": str(path), "sha256": sha256_file(path), "rows": count})
    if not rows:
        raise ValueError("no counterfactual action-Q rows")
    return rows, audits


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes = sorted({str(row["episode_id"]) for row in rows})
    validation_episodes = {episode for index, episode in enumerate(episodes) if index % 5 == 0}
    train = [row for row in rows if str(row["episode_id"]) not in validation_episodes]
    validation = [row for row in rows if str(row["episode_id"]) in validation_episodes]
    return (train or list(rows), validation or list(rows[: min(256, len(rows))]))


def selected_values(actor, q_model, batch: dict[str, Any]) -> torch.Tensor:
    with torch.no_grad():
        state, encoded_options, _ = actor.encode_batch(batch)
    values = q_model(state, encoded_options)
    indices = torch.tensor(
        [int(row["option_index"]) for row in batch["rows"]], dtype=torch.long, device=values.device
    )
    return values[torch.arange(values.shape[0], device=values.device), indices]


@torch.inference_mode()
def validate(actor, q_model, rows: list[dict[str, Any]], batch_size: int, device: torch.device) -> dict[str, float]:
    q_model.eval()
    absolute = squared = uncertainty = 0.0
    count = 0
    loader = make_loader(TrajectoryDataset(rows), batch_size, shuffle=False, seed=0)
    for original in loader:
        batch = to_device(original, device)
        values = selected_values(actor, q_model, batch)
        target = torch.tensor([float(row["q_target"]) for row in original["rows"]], device=device)
        mean = values.mean(dim=1)
        absolute += float((mean - target).abs().sum())
        squared += float((mean - target).square().sum())
        uncertainty += float(values.std(dim=1, unbiased=False).sum())
        count += len(original["rows"])
    return {"rows": count, "mae": absolute / count, "rmse": (squared / count) ** 0.5, "mean_uncertainty": uncertainty / count}


def atomic_save(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train counterfactual action-conditioned Q(s,a) ensemble")
    parser.add_argument("--rows", nargs="+", required=True, type=Path)
    parser.add_argument("--actor-checkpoint", required=True, type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.metrics_output.exists():
        raise FileExistsError("refusing to overwrite action-Q output")
    if not args.allow_dirty_smoke:
        assert_clean_worktree()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    started = time.perf_counter()
    actor, actor_payload = load_checkpoint(args.actor_checkpoint, device)
    actor.eval()
    actor.requires_grad_(False)
    q_model = ActionValueEnsemble(int(actor_payload["hidden_dim"]), args.heads).to(device)
    if args.initialize_from and args.initialize_from.is_file():
        previous = torch.load(args.initialize_from, map_location=device, weights_only=False)
        q_model.load_state_dict(previous["model"])
    rows, audits = load_rows(args.rows)
    train_rows, validation_rows = split_rows(rows)
    optimizer = torch.optim.Adam(q_model.parameters(), lr=args.learning_rate)
    epochs = []
    for epoch in range(1, args.epochs + 1):
        q_model.train()
        loss_sum = 0.0
        batches = 0
        loader = make_loader(TrajectoryDataset(train_rows), args.batch_size, shuffle=True, seed=args.seed + epoch)
        for original in loader:
            batch = to_device(original, device)
            values = selected_values(actor, q_model, batch)
            targets = torch.tensor([float(row["q_target"]) for row in original["rows"]], device=device)
            per_head = F.smooth_l1_loss(values, targets[:, None].expand_as(values), reduction="none")
            mask = (torch.rand_like(per_head) < 0.8).to(per_head.dtype)
            loss = (per_head * mask).sum() / mask.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            batches += 1
        record = {"epoch": epoch, "loss": loss_sum / max(1, batches), "batches": batches}
        epochs.append(record)
        print(json.dumps(record), flush=True)
    validation = validate(actor, q_model, validation_rows, args.batch_size, device)
    payload = {
        "schema_version": 1,
        "kind": "counterfactual_action_q_ensemble",
        "model": q_model.state_dict(),
        "hidden_dim": int(actor_payload["hidden_dim"]),
        "heads": args.heads,
        "actor_checkpoint_sha256": sha256_file(args.actor_checkpoint),
        "training_rows": audits,
        "epochs": epochs,
        "validation": validation,
    }
    atomic_save(payload, args.output)
    metrics = {
        "status": "completed_formal" if not args.allow_dirty_smoke else "completed_smoke",
        "rows": len(rows), "train_rows": len(train_rows), "validation_rows": len(validation_rows),
        "validation": validation, "epochs": epochs, "device": str(device),
        "actor_checkpoint_sha256": payload["actor_checkpoint_sha256"],
        "q_checkpoint_sha256": sha256_file(args.output),
        "runtime_seconds": time.perf_counter() - started,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
