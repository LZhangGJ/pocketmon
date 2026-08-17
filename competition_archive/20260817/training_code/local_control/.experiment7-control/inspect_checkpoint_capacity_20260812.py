#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    tensors = {key: value for key, value in state.items() if hasattr(value, "numel")}
    total = sum(value.numel() for value in tensors.values())
    trainable_like = sum(
        value.numel()
        for key, value in tensors.items()
        if not key.endswith("num_batches_tracked")
    )
    largest = sorted(
        ((key, value.numel(), list(value.shape)) for key, value in tensors.items()),
        key=lambda row: row[1],
        reverse=True,
    )[:12]
    result = {
        "checkpoint": str(args.checkpoint),
        "bytes": args.checkpoint.stat().st_size,
        "parameter_tensors": len(tensors),
        "parameters": total,
        "parameters_excluding_counters": trainable_like,
        "model_config": payload.get("model_config"),
        "config": payload.get("config"),
        "largest_tensors": largest,
        "top_level_keys": sorted(payload.keys()) if isinstance(payload, dict) else None,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
