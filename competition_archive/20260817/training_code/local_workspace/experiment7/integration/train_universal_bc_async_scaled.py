from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

import train_universal_bc as core
from common import Experiment7Error, read_json, utc_now, write_json


def atomic_checkpoint(path: Path, model: torch.nn.Module, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    core.save_checkpoint(temporary, model, metadata)
    os.replace(temporary, path)


def pending_count(queue: Path, results: Path) -> int:
    return sum(1 for job in queue.glob("epoch_*.json") if not (results / job.name).is_file())


def scaled_model_config_kwargs(sources: dict, args: argparse.Namespace) -> dict:
    card_vocab = int(sources["engineCatalog"]["cardVocab"])
    expected = args.expected_card_vocab
    if expected is not None and card_vocab != expected:
        raise Experiment7Error(
            f"source cardVocab {card_vocab} does not match expected {expected}"
        )
    return {
        "card_vocab": card_vocab,
        "d_model": args.d_model,
        "n_heads": args.heads,
        "n_layers": args.layers,
        "ff_dim": args.ff_dim,
        "dropout": args.dropout,
    }


def validate_source_contract(sources: dict) -> None:
    kind = sources.get("kind")
    if kind == "experiment7_universal_bc":
        return
    if kind == "experiment7_universal_bc_strict_score_window":
        if sources.get("minGameScoreExclusive") != 1000.0:
            raise Experiment7Error("strict Universal BC source threshold must be >1000")
        if sources.get("policySource") != "winners":
            raise Experiment7Error("strict Universal BC policy source must be winners")
        return
    raise Experiment7Error("scaled async trainer requires Universal BC sources")


def initialize_scaled_model(
    model: torch.nn.Module,
    checkpoint: Path | None,
    device: torch.device,
) -> dict | None:
    if checkpoint is None:
        return None
    checkpoint = checkpoint.resolve()
    payload = core.load_checkpoint(checkpoint, device)
    expected_config = model.config.to_dict()
    actual_config = payload.get("config")
    if actual_config != expected_config:
        raise Experiment7Error(
            "scaled initialization checkpoint config does not match target config"
        )
    model.load_state_dict(payload["state_dict"], strict=True)
    return {
        "path": str(checkpoint),
        "sha256": core.sha256_file(checkpoint),
        "strict": True,
    }


def train(args: argparse.Namespace) -> dict:
    sources = read_json(args.sources.resolve())
    validate_source_contract(sources)
    vendor = core.setup_vendor(Path(sources["referenceRoot"]))
    core.seed_everything(args.seed)
    device = core.device_from_arg(args.device)
    shards = core.prepare_shards(sources, vendor, args.validation_fraction, 0, 0)
    config = vendor["UniversalDeckModelConfig"](
        **scaled_model_config_kwargs(sources, args)
    )
    model = vendor["UniversalDeckTransformerPolicy"](config).to(device)
    initialization = initialize_scaled_model(model, args.initialize_from, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule = sorted(
        (int(epoch_text), float(rate_text))
        for epoch_text, rate_text in (value.split("=", 1) for value in args.learning_rate_schedule)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(args.seed)

    output = args.output_dir.resolve()
    queue, results, control = output / "validation-queue", output / "validation-results", output / "control"
    for path in (output, queue, results, control, output / "checkpoints"):
        path.mkdir(parents=True, exist_ok=True)
    report_path = output / "async-training-report.json"
    report = {
        "schemaVersion": 1,
        "mode": "scaled_persistent_training_with_async_validation",
        "createdAt": utc_now(),
        "sources": str(args.sources.resolve()),
        "device": str(device),
        "config": config.to_dict(),
        "parameterCount": model.parameter_count,
        "initialization": initialization,
        "maxPendingValidations": args.max_pending_validations,
        "epochs": [],
    }
    write_json(report_path, report)
    print(
        {
            "stage": "scaled_async_train_start",
            "parameterCount": model.parameter_count,
            "initialization": initialization,
        },
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        while pending_count(queue, results) >= args.max_pending_validations:
            if (control / "stop-request.json").is_file():
                report["stop"] = read_json(control / "stop-request.json")
                report["completedAt"] = utc_now()
                write_json(report_path, report)
                return report
            time.sleep(args.poll_seconds)
        if (control / "stop-request.json").is_file():
            report["stop"] = read_json(control / "stop-request.json")
            report["completedAt"] = utc_now()
            write_json(report_path, report)
            return report

        learning_rate = args.learning_rate
        for milestone, rate in schedule:
            if epoch >= milestone:
                learning_rate = rate
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        rows = []
        for shard_index in rng.permutation(len(shards)):
            shard = shards[int(shard_index)]
            metrics = core.train_epoch(
                vendor, model, shard["bundle"], shard["train"], shard["policyWeights"],
                optimizer, scaler, device, args.batch_size, rng, args.value_loss_weight,
                args.prefetch_batches, args.prefetch_workers,
            )
            rows.append((shard["name"], metrics))
            print({"stage": "train_shard", "epoch": epoch, "shard": shard["name"], **metrics}, flush=True)
        checkpoint = output / "checkpoints" / f"epoch_{epoch:06d}.pt"
        row = {
            "epoch": epoch,
            "learningRate": learning_rate,
            "training": core.combine_training_metrics(rows),
            "checkpoint": str(checkpoint),
            "publishedAt": utc_now(),
            "validationStatus": "pending",
        }
        atomic_checkpoint(checkpoint, model, row)
        write_json(queue / f"epoch_{epoch:06d}.json", row)
        report["epochs"].append(row)
        write_json(report_path, report)
        print({"stage": "checkpoint_published", **row}, flush=True)
    report["completedAt"] = utc_now()
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--epochs", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--prefetch-batches", type=int, default=6)
    parser.add_argument("--prefetch-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--learning-rate-schedule", action="append", default=[])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--max-pending-validations", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--expected-card-vocab", type=int)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
