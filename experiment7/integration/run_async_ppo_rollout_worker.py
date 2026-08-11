from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from async_ppo_control import atomic_write_json, read_json, utc_now


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously collect asynchronous PPO shards on CPU")
    parser.add_argument("--league", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--buffer-root", type=Path, required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--worker-id", default=socket.gethostname())
    parser.add_argument("--episodes-per-shard", type=int, default=20)
    parser.add_argument("--refresh-rounds", type=int, default=1)
    parser.add_argument("--self-play-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if args.episodes_per_shard <= 0 or args.refresh_rounds <= 0:
        raise ValueError("shard and refresh sizes must be positive")
    collector = args.worktree / "experiment7/integration/collect_universal_ppo_rollouts.py"
    sequence = 0
    env = dict(os.environ)
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    while True:
        league = read_json(args.league)
        pool = Path(league["poolPath"])
        for _ in range(args.refresh_rounds):
            for chain_name, chain in sorted(league["chains"].items()):
                current = chain["current"]
                sequence += 1
                stamp = time.time_ns()
                shard_id = f"{args.worker_id}-{sequence:08d}-{stamp}"
                output = args.buffer_root / "ready" / chain_name / f"{shard_id}.jsonl.gz"
                output.parent.mkdir(parents=True, exist_ok=True)
                log = args.buffer_root / "logs" / args.worker_id / f"{shard_id}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                seed = (stamp ^ hash((args.worker_id, chain_name, sequence))) & 0x7FFFFFFF
                command = [
                    args.python,
                    str(collector),
                    "--reference-root",
                    str(Path(league["referenceRoot"])),
                    "--engine-catalog",
                    str(Path(league["engineCatalog"])),
                    "--checkpoint",
                    str(Path(current["checkpoint"])),
                    "--teacher",
                    str(Path(chain["teacher"])),
                    "--deck",
                    str(Path(chain["deckPath"])),
                    "--pool",
                    str(pool),
                    "--cg-dir",
                    str(Path(league["cgDir"])),
                    "--episodes",
                    str(args.episodes_per_shard),
                    "--self-play-fraction",
                    str(args.self_play_fraction),
                    "--temperature",
                    "1.0",
                    "--max-decisions",
                    "5000",
                    "--seed",
                    str(seed),
                    "--run-id",
                    shard_id,
                    "--behavior-generation",
                    str(current["generation"]),
                    "--behavior-snapshot-id",
                    str(current["snapshotId"]),
                    "--role",
                    "diversity",
                    "--device",
                    "cpu",
                    "--output",
                    str(output),
                ]
                started = time.perf_counter()
                with log.open("w", encoding="utf-8") as handle:
                    completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
                if completed.returncode:
                    atomic_write_json(
                        output.with_suffix(output.suffix + ".failed.json"),
                        {
                            "createdAt": utc_now(),
                            "worker": args.worker_id,
                            "chain": chain_name,
                            "snapshotId": current["snapshotId"],
                            "returnCode": completed.returncode,
                            "log": str(log.resolve()),
                            "seconds": time.perf_counter() - started,
                        },
                    )
                    continue
        # Refresh all three policy checkpoints and the dynamic pool together.


if __name__ == "__main__":
    main()
