#!/usr/bin/env python3
"""Create isolated one-chain league roots for historical A08 evaluations.

The generated league states point at already-published deployment packages and
never mutate the live Experiment 7 league.  Each root can therefore use the
standard full-matrix runner and its load-guarded distributed Arena machinery.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


CHAIN = "a08_dipplin_seaking"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def record_for(chain: dict, generation: int, live_root: Path) -> dict:
    records = [chain.get("current", {}), *chain.get("history", [])]
    for record in records:
        if int(record.get("generation", -1)) == generation:
            return copy.deepcopy(record)
    generation_root = live_root / "learners" / CHAIN / f"generation-{generation:06d}"
    return {
        "generation": generation,
        "snapshotId": f"{CHAIN}-g{generation:06d}-historical-checkpoint",
        "checkpoint": str(generation_root / "checkpoint.pt"),
        "metrics": str(generation_root / "metrics.json"),
        "packageManifest": str(generation_root / "deployment" / "packages" / "packages.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--live-root", type=Path, required=True)
    parser.add_argument("--generations", type=int, nargs="+", default=[253, 277, 294, 333])
    args = parser.parse_args()

    source = read_json(args.league)
    chain = source["chains"][CHAIN]
    receipt = {"schemaVersion": 1, "chain": CHAIN, "candidates": []}
    for generation in args.generations:
        candidate = copy.deepcopy(source)
        selected_chain = copy.deepcopy(chain)
        selected = record_for(chain, generation, args.live_root)
        selected_chain["current"] = selected
        selected_chain["history"] = []
        candidate["chains"] = {CHAIN: selected_chain}
        candidate.pop("trainingControlUpdatedAt", None)
        candidate.pop("trainingControlSourceRoundId", None)
        root = args.output_root / f"g{generation:06d}"
        write_json(root / "state" / "league.json", candidate)
        receipt["candidates"].append(
            {
                "generation": generation,
                "root": str(root),
                "checkpoint": selected["checkpoint"],
                "packageManifest": selected["packageManifest"],
            }
        )
    write_json(args.output_root / "candidates.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
