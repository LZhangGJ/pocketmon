from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.features import ACTION_DIM
from rl.model import CandidateActorCritic


class TrajectoryDataset(Dataset):
    def __init__(self, path: Path) -> None:
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        self.rows = [row for row in self.rows if len(row["chosen"]) == 1 and row["options"]]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return row["state"], row["options"], row["chosen"][0], row["reward"]


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior-clone a candidate policy with outcome-weighted value learning")
    parser.add_argument("--input", default="data/rl/trajectories.jsonl")
    parser.add_argument("--output", default="artifacts/rl/candidate_actor_critic.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()
    dataset = TrajectoryDataset(Path(args.input))
    if not dataset:
        raise ValueError("no single-option decisions found in the trajectory file")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    model = CandidateActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        total = 0.0
        for states, options, chosen, rewards, mask in loader:
            logits, values = model(states.float(), options.float())
            logits = logits.masked_fill(~mask.bool(), -1e9)
            policy_loss = F.cross_entropy(logits, chosen.long())
            value_loss = F.mse_loss(values, rewards.float())
            loss = policy_loss + 0.25 * value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(states)
        print(json.dumps({"epoch": epoch + 1, "loss": total / len(dataset)}))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "state_dim": 32, "action_dim": ACTION_DIM}, output)
    print(output)


if __name__ == "__main__":
    main()
