from __future__ import annotations

import argparse
import json
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


def atomic_progress_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    rng: np.random.Generator,
    progress: dict,
) -> None:
    """Persist enough state to resume inside an epoch after a completed day shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "schema_version": 3,
            "architecture": "experiment7_universal_deck8_autoregressive_stop",
            "config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "progress": progress,
        },
        temporary,
    )
    os.replace(temporary, path)


def pending_count(queue: Path, results: Path) -> int:
    return sum(
        1
        for job in queue.glob("epoch_*.json")
        if not (results / job.name).is_file()
    )


def stop_requested(control: Path) -> dict | None:
    path = control / "stop-request.json"
    return read_json(path) if path.is_file() else None


def train(args: argparse.Namespace) -> dict:
    sources = read_json(args.sources.resolve())
    if sources.get("kind") != "experiment7_universal_bc":
        raise Experiment7Error("async trainer requires Universal BC sources")
    vendor = core.setup_vendor(Path(sources["referenceRoot"]))
    core.seed_everything(args.seed)
    device = core.device_from_arg(args.device)
    shards = core.prepare_shards(
        sources, vendor, args.validation_fraction, args.max_train_decisions, 0
    )
    resume_payload = (
        core.load_checkpoint(args.resume_progress.resolve(), device)
        if args.resume_progress is not None
        else None
    )
    payload = resume_payload or core.load_checkpoint(args.initialize_from.resolve(), device)
    config = vendor["UniversalDeckModelConfig"](**payload["config"])
    model = vendor["UniversalDeckTransformerPolicy"](config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    schedule = []
    for value in args.learning_rate_schedule:
        epoch_text, rate_text = value.split("=", 1)
        schedule.append((int(epoch_text), float(rate_text)))
    schedule.sort()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(args.seed)
    resume = None
    if resume_payload is not None:
        resume = resume_payload.get("progress")
        if not isinstance(resume, dict):
            raise Experiment7Error("resume checkpoint lacks progress metadata")
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scaler.load_state_dict(resume_payload.get("scaler_state_dict", {}))
        rng.bit_generator.state = resume_payload["numpy_rng_state"]
        torch.set_rng_state(resume_payload["torch_rng_state"].cpu())
        if device.type == "cuda" and resume_payload.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state_all"])

    output = args.output_dir.resolve()
    queue = output / "validation-queue"
    results = output / "validation-results"
    control = output / "control"
    for path in (output, queue, results, control, output / "checkpoints"):
        path.mkdir(parents=True, exist_ok=True)
    report_path = output / "async-training-report.json"
    report = {
        "schemaVersion": 1,
        "mode": "persistent_training_with_async_validation",
        "createdAt": utc_now(),
        "sources": str(args.sources.resolve()),
        "initialization": str(args.initialize_from.resolve()),
        "resumedFrom": str(args.resume_progress.resolve()) if args.resume_progress else None,
        "device": str(device),
        "parameterCount": model.parameter_count,
        "maxPendingValidations": args.max_pending_validations,
        "epochs": [],
    }
    write_json(report_path, report)
    print(json.dumps({"stage": "async_train_start", **report}, ensure_ascii=False), flush=True)

    for epoch in range(args.epoch_start, args.epoch_start + args.epochs):
        while pending_count(queue, results) >= args.max_pending_validations:
            stop = stop_requested(control)
            if stop:
                report["stop"] = stop
                report["completedAt"] = utc_now()
                write_json(report_path, report)
                return report
            time.sleep(args.poll_seconds)
        stop = stop_requested(control)
        if stop:
            report["stop"] = stop
            report["completedAt"] = utc_now()
            write_json(report_path, report)
            return report

        learning_rate = args.learning_rate
        for milestone_epoch, milestone_rate in schedule:
            if epoch >= milestone_epoch:
                learning_rate = milestone_rate
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        if resume is not None:
            if int(resume["epoch"]) != epoch:
                raise Experiment7Error("resume checkpoint epoch does not match --epoch-start")
            shard_order = [int(value) for value in resume["shardOrder"]]
            next_position = int(resume["nextPosition"])
            rows = [(str(name), metrics) for name, metrics in resume.get("rows", [])]
            resume = None
        else:
            shard_order = [int(value) for value in rng.permutation(len(shards))]
            next_position = 0
            rows = []
        for position in range(next_position, len(shard_order)):
            shard_index = shard_order[position]
            shard = shards[int(shard_index)]
            metrics = core.train_epoch(
                vendor, model, shard["bundle"], shard["train"],
                shard["policyWeights"], optimizer, scaler, device,
                args.batch_size, rng, args.value_loss_weight,
                args.prefetch_batches, args.prefetch_workers,
            )
            rows.append((shard["name"], metrics))
            print(json.dumps({"stage": "train_shard", "epoch": epoch, "shard": shard["name"], **metrics}), flush=True)
            progress_path = output / "recovery" / f"epoch_{epoch:06d}_latest.pt"
            atomic_progress_checkpoint(
                progress_path,
                model,
                optimizer,
                scaler,
                rng,
                {
                    "epoch": epoch,
                    "shardOrder": shard_order,
                    "nextPosition": position + 1,
                    "rows": rows,
                    "completedShard": shard["name"],
                    "publishedAt": utc_now(),
                },
            )
            print(json.dumps({"stage": "recovery_checkpoint_published", "epoch": epoch, "shard": shard["name"], "checkpoint": str(progress_path)}), flush=True)
        training = core.combine_training_metrics(rows)
        checkpoint = output / "checkpoints" / f"epoch_{epoch:06d}.pt"
        epoch_row = {
            "epoch": epoch,
            "learningRate": learning_rate,
            "training": training,
            "checkpoint": str(checkpoint),
            "publishedAt": utc_now(),
            "validationStatus": "pending",
        }
        atomic_checkpoint(checkpoint, model, epoch_row)
        write_json(queue / f"epoch_{epoch:06d}.json", epoch_row)
        report["epochs"].append(epoch_row)
        write_json(report_path, report)
        print(json.dumps({"stage": "checkpoint_published", **epoch_row}), flush=True)
    report["completedAt"] = utc_now()
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initialize-from", type=Path, required=True)
    parser.add_argument("--resume-progress", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--epochs", type=int, default=1_000_000)
    parser.add_argument("--epoch-start", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prefetch-batches", type=int, default=6)
    parser.add_argument("--prefetch-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--learning-rate-schedule", action="append", default=[])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=0.05)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--max-train-decisions", type=int, default=0)
    parser.add_argument("--max-pending-validations", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    if args.max_pending_validations < 1:
        parser.error("--max-pending-validations must be positive")
    train(args)


if __name__ == "__main__":
    main()
