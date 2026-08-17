#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHAINS = {
    "a02_submission4_grimmsnarl_froslass_munkidori": "A02",
    "a08_dipplin_seaking": "A08",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resource_snapshot() -> tuple[float, float]:
    load = float(Path("/proc/loadavg").read_text(encoding="utf-8").split()[0])
    cpu = 100.0 * load / max(os.cpu_count() or 1, 1)
    io = 0.0
    pressure = Path("/proc/pressure/io")
    if pressure.is_file():
        for token in pressure.read_text(encoding="utf-8").splitlines()[0].split():
            if token.startswith("avg10="):
                io = float(token.split("=", 1)[1])
                break
    return cpu, io


def wait_for_capacity(limit: float, poll_seconds: float) -> tuple[float, float]:
    while True:
        cpu, io = resource_snapshot()
        if cpu < limit and io < limit:
            return cpu, io
        print(f"PACKAGE_GUARD_WAIT cpu={cpu:.2f}% io_pressure={io:.2f}%", flush=True)
        time.sleep(poll_seconds)


def copy_agent(source: Path, destination: Path, limit: float, poll: float) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if "__pycache__" in relative.parts or path.name.endswith((".pyc", ".pyo")):
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            wait_for_capacity(limit, poll)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze A02 and A08 as unpacked Kaggle agent directories")
    parser.add_argument("--league-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resource-limit", type=float, default=70.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()

    league_root = args.league_root.resolve()
    league = read_json(league_root / "state/league.json")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = args.output_root.resolve() / f"a02-a08-kaggle-agents-unpacked-{stamp}"
    destination.mkdir(parents=True, exist_ok=False)

    receipts: list[dict[str, Any]] = []
    for chain, archetype_id in CHAINS.items():
        current = dict(league["chains"][chain]["current"])
        generation = int(current["generation"])
        manifest_path = Path(current["packageManifest"]).resolve()
        manifest = read_json(manifest_path)
        packages = [
            row for row in manifest["packages"]
            if str(row.get("archetypeId", "")).upper() == archetype_id
        ]
        if len(packages) != 1:
            raise RuntimeError(f"expected one {archetype_id} package, found {len(packages)}")
        source = Path(packages[0]["agentDir"]).resolve()
        for required in (source / "main.py", source / "deck.csv"):
            if not required.is_file():
                raise FileNotFoundError(required)

        name = f"{archetype_id.lower()}_g{generation:06d}_kaggle_agent"
        target = destination / name
        cpu, io = wait_for_capacity(args.resource_limit, args.poll_seconds)
        print(
            f"PACKAGE_START chain={chain} generation={generation} "
            f"cpu={cpu:.2f}% io_pressure={io:.2f}%",
            flush=True,
        )
        copy_agent(source, target, args.resource_limit, args.poll_seconds)
        files = sorted(path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file())
        receipts.append(
            {
                "chain": chain,
                "archetypeId": archetype_id,
                "generation": generation,
                "snapshotId": current.get("snapshotId"),
                "checkpoint": current.get("checkpoint"),
                "sourceAgentDir": str(source),
                "outputAgentDir": str(target),
                "files": files,
                "requiredFilesPresent": all((target / name).is_file() for name in ("main.py", "deck.csv")),
            }
        )

    receipt = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "format": "unpacked Kaggle agent directories",
        "archiveCreated": False,
        "uploadedToKaggle": False,
        "hashVerificationSkipped": True,
        "resourceLimitPercent": args.resource_limit,
        "outputRoot": str(destination),
        "agents": receipts,
    }
    receipt_path = destination / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
