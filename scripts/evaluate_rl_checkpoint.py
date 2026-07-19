from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_rl_policy import TrajectoryDataset, choose_device, make_loader, run_epoch

from rl.model import CandidateActorCritic


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an RL checkpoint on held-out trajectory JSONL")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = CandidateActorCritic().to(device)
    model.load_state_dict(checkpoint["model"])
    dataset = TrajectoryDataset.from_path(Path(args.input))
    if not dataset:
        raise ValueError("no eligible decisions found")
    metrics = run_epoch(model, make_loader(dataset, args.batch_size, False, 0), device)
    print(json.dumps({"checkpoint": args.checkpoint, "device": str(device), **metrics}))


if __name__ == "__main__":
    main()
