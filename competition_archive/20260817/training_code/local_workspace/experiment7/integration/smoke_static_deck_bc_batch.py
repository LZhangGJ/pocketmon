from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import train_universal_bc as core
from common import sha256_file, utc_now, write_json
from static_deck_bc_common import load_json
from train_static_deck_bc_async import validate_static_sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    sources = validate_static_sources(args.sources.resolve(), config)
    vendor = core.setup_vendor(Path(sources["referenceRoot"]))
    core.seed_everything(20260815)
    device = core.device_from_arg(args.device)
    shards = core.prepare_shards(sources, vendor, 0.05, 0, 0)
    payload = core.load_checkpoint(args.checkpoint.resolve(), device)
    model_config = vendor["UniversalDeckModelConfig"](**payload["config"])
    model = vendor["UniversalDeckTransformerPolicy"](model_config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    shard = shards[0]
    decisions = shard["train"][: args.batch_size]
    batch = vendor["make_identity_batch"](shard["bundle"], decisions, device)
    weights = torch.from_numpy(shard["policyWeights"][decisions].astype(np.float32, copy=True)).to(device)
    values = torch.ones(len(decisions), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        encoding = core.forward_model(model, batch)
        loss, parts = vendor["universal_bc_loss"](
            model,
            encoding,
            batch,
            weights,
            values,
            value_loss_weight=0.05,
        )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    receipt = {
        "schemaVersion": 1,
        "kind": "experiment7_static_deck_bc_minibatch_smoke",
        "completedAt": utc_now(),
        "sources": str(args.sources.resolve()),
        "sourcesSha256": sha256_file(args.sources.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpointSha256": sha256_file(args.checkpoint.resolve()),
        "strictPredicate": config["strictPredicate"],
        "profile": sources["staticProfile"],
        "device": str(device),
        "gpuName": torch.cuda.get_device_name(device),
        "torchVersion": torch.__version__,
        "cudaVersion": torch.version.cuda,
        "parameterCount": int(model.parameter_count),
        "batchSize": int(len(decisions)),
        "positivePolicyWeights": int(torch.count_nonzero(weights).item()),
        "loss": float(loss.detach().cpu()),
        "lossParts": {name: float(value.detach().cpu()) for name, value in parts.items()},
        "seconds": elapsed,
        "peakMemoryBytes": int(torch.cuda.max_memory_allocated(device)),
        "forwardBackwardOptimizerStep": True,
        "checkpointWritten": False,
        "formalEpochStarted": False,
    }
    write_json(args.output.resolve(), receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
