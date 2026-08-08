from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an Experiment 7 torch checkpoint to NumPy-only portable NPZ")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    state = checkpoint.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError("checkpoint must contain config and state_dict")
    arrays: dict[str, np.ndarray] = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state_dict value is not a tensor: {name}")
        arrays[name] = value.detach().cpu().numpy().astype(np.float32, copy=False)
    arrays["config_json"] = np.asarray(
        [json.dumps(config, sort_keys=True, separators=(",", ":"))], dtype=np.str_
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    receipt = args.receipt or args.output.with_suffix(args.output.suffix + ".receipt.json")
    write_json(
        receipt,
        {
            "schema_version": 1,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "portable": str(args.output.resolve()),
            "portable_sha256": sha256_file(args.output),
            "array_count": len(arrays) - 1,
            "config": config,
            "optimizer_included": False,
        },
    )
    print(receipt)


if __name__ == "__main__":
    main()
