from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.features import ACTION_DIM, STATE_DIM
from rl.model import CandidateActorCritic


class TrajectoryDataset(Dataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = [row for row in rows if len(row["chosen"]) == 1 and row["options"]]

    @classmethod
    def from_path(cls, path: Path) -> "TrajectoryDataset":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        return cls(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return row["state"], row["options"], row["chosen"][0], row["reward"]


def split_by_episode(dataset: TrajectoryDataset, validation_fraction: float, seed: int):
    """Keep every decision from an episode in one split to prevent leakage."""
    episodes = sorted({int(row["episode"]) for row in dataset.rows})
    if validation_fraction <= 0 or len(episodes) < 2:
        return dataset, TrajectoryDataset([])
    rng = random.Random(seed)
    rng.shuffle(episodes)
    validation_count = max(1, min(len(episodes) - 1, round(len(episodes) * validation_fraction)))
    validation_episodes = set(episodes[:validation_count])
    train = [row for row in dataset.rows if int(row["episode"]) not in validation_episodes]
    validation = [row for row in dataset.rows if int(row["episode"]) in validation_episodes]
    return TrajectoryDataset(train), TrajectoryDataset(validation)


def collate(rows):
    max_options = max(len(row[1]) for row in rows)
    states, options, chosen, rewards, masks = [], [], [], [], []
    for state, row_options, action, reward in rows:
        padding = [[0.0] * ACTION_DIM] * (max_options - len(row_options))
        states.append(state)
        options.append(row_options + padding)
        chosen.append(action)
        rewards.append(reward)
        masks.append([True] * len(row_options) + [False] * len(padding))
    return tuple(torch.tensor(value) for value in (states, options, chosen, rewards, masks))


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(dataset: TrajectoryDataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, loader, device, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "correct": 0.0, "samples": 0.0}
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for states, options, chosen, rewards, mask in loader:
            states = states.to(device, dtype=torch.float32, non_blocking=True)
            options = options.to(device, dtype=torch.float32, non_blocking=True)
            chosen = chosen.to(device, dtype=torch.long, non_blocking=True)
            rewards = rewards.to(device, dtype=torch.float32, non_blocking=True)
            mask = mask.to(device, dtype=torch.bool, non_blocking=True)
            logits, values = model(states, options)
            logits = logits.masked_fill(~mask, -torch.inf)
            policy_loss = F.cross_entropy(logits, chosen)
            value_loss = F.mse_loss(values, rewards)
            loss = policy_loss + 0.25 * value_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            count = len(states)
            totals["loss"] += float(loss.detach()) * count
            totals["policy_loss"] += float(policy_loss.detach()) * count
            totals["value_loss"] += float(value_loss.detach()) * count
            totals["correct"] += float((logits.argmax(dim=1) == chosen).sum())
            totals["samples"] += count
    samples = totals.pop("samples")
    correct = totals.pop("correct")
    return {**{name: value / samples for name, value in totals.items()}, "accuracy": correct / samples, "samples": int(samples)}


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
            "config": vars(args),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior-clone a candidate policy with outcome value learning")
    parser.add_argument("--input", default="data/rl/trajectories.jsonl")
    parser.add_argument("--output", default="artifacts/rl/candidate_actor_critic.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args()
    if not 0 <= args.validation_fraction < 1:
        raise ValueError("validation-fraction must be in [0, 1)")
    seed_everything(args.seed)
    device = choose_device(args.device)
    dataset = TrajectoryDataset.from_path(Path(args.input))
    train, validation = split_by_episode(dataset, args.validation_fraction, args.seed)
    if not train:
        raise ValueError("no single-option training decisions found")
    train_loader = make_loader(train, args.batch_size, True, args.seed)
    validation_loader = make_loader(validation, args.batch_size, False, args.seed) if validation else None
    model = CandidateActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output = Path(args.output)
    best_output = output.with_name(f"{output.stem}.best{output.suffix}")
    best_loss = float("inf")
    print(json.dumps({"device": str(device), "train_samples": len(train), "validation_samples": len(validation)}))
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        validation_metrics = run_epoch(model, validation_loader, device) if validation_loader else train_metrics
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        print(json.dumps(record))
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            save_checkpoint(best_output, model, optimizer, epoch, record, args)
    save_checkpoint(output, model, optimizer, args.epochs, record, args)
    print(json.dumps({"last": str(output), "best": str(best_output), "best_validation_loss": best_loss}))


if __name__ == "__main__":
    main()
