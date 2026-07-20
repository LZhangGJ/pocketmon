from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.bc import TrajectoryDataset, build_split_manifest, evaluate_decoding, load_replay_dataset, make_loader, run_epoch
from rl.model import MaskedPointerActorCritic
from scripts.train_rl_policy import choose_device


def validate_input_sha(checkpoint: dict, actual_sha256: str, allow_mismatch: bool = False) -> None:
    expected = checkpoint.get("input_sha256")
    if expected != actual_sha256 and not allow_mismatch:
        raise ValueError(
            f"input SHA-256 mismatch: checkpoint={expected!r}, current={actual_sha256!r}; "
            "pass --allow-input-mismatch only for an explicitly audited comparison"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an RL-BC-001 checkpoint on the fixed episode validation split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", default="data/processed/public_replay_v1.jsonl.gz")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-input-mismatch", action="store_true")
    args = parser.parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    rows, audit = load_replay_dataset(Path(args.input))
    validate_input_sha(checkpoint, audit["input_sha256"], args.allow_input_mismatch)
    config = checkpoint["config"]
    _, _, validation_ids = build_split_manifest(rows, config["validation_fraction"], config["split_seed"])
    validation = TrajectoryDataset([row for row in rows if row["episode_id"] in validation_ids])
    loader = make_loader(validation, args.batch_size, False, config["seed"])
    model = MaskedPointerActorCritic(checkpoint["hidden_dim"]).to(device)
    model.load_state_dict(checkpoint["model"])
    result = {
        "checkpoint": args.checkpoint, "checkpoint_git_sha": checkpoint["git_sha"], "device": str(device),
        "dataset_sha256": audit["input_sha256"], "validation_rows": len(validation),
        "loss": run_epoch(model, loader, device), "validation": evaluate_decoding(model, loader, device),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
