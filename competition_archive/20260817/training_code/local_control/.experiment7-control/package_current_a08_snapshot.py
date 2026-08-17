from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHAIN = "a08_dipplin_seaking"


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
        print(f"A08_PACKAGE_GUARD_WAIT cpu={cpu:.2f}% io_pressure={io:.2f}%", flush=True)
        time.sleep(poll_seconds)


def add_file(archive: tarfile.TarFile, source: Path, arcname: str, limit: float, poll: float) -> None:
    wait_for_capacity(limit, poll)
    archive.add(source, arcname=arcname, recursive=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze current A08 games and Kaggle-ready agent")
    parser.add_argument("--league-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resource-limit", type=float, default=70.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    args = parser.parse_args()

    league_root = args.league_root.resolve()
    league = read_json(league_root / "state/league.json")
    current = dict(league["chains"][CHAIN]["current"])
    generation = int(current["generation"])
    manifest_path = Path(current["packageManifest"]).resolve()
    package_manifest = read_json(manifest_path)
    packages = [row for row in package_manifest["packages"] if str(row.get("archetypeId", "")).upper() == "A08"]
    if len(packages) != 1:
        raise RuntimeError(f"expected one A08 package, found {len(packages)}")
    agent_dir = Path(packages[0]["agentDir"]).resolve()
    for required in (agent_dir / "main.py", agent_dir / "deck.csv"):
        if not required.is_file():
            raise FileNotFoundError(required)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = args.output_root.resolve() / f"a08-g{generation:06d}-{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    cpu, io = wait_for_capacity(args.resource_limit, args.poll_seconds)
    print(
        f"A08_PACKAGE_START generation={generation} cpu={cpu:.2f}% io_pressure={io:.2f}%",
        flush=True,
    )

    agent_zip = destination / f"a08-g{generation:06d}-kaggle-agent.zip"
    with zipfile.ZipFile(agent_zip, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(agent_dir.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                wait_for_capacity(args.resource_limit, args.poll_seconds)
                archive.write(path, path.relative_to(agent_dir).as_posix())

    ready = league_root / "buffer/ready" / CHAIN
    summaries = sorted(ready.glob("*.jsonl.gz.summary.json"))
    rollout_pairs: list[tuple[Path, Path]] = []
    for summary in summaries:
        payload = read_json(summary)
        rollout = Path(payload["output"]["path"]).resolve()
        if rollout.is_file():
            rollout_pairs.append((rollout, summary))

    games_tar = destination / f"a08-g{generation:06d}-games.tar"
    with tarfile.open(games_tar, "x") as archive:
        for index, (rollout, summary) in enumerate(rollout_pairs):
            add_file(archive, rollout, f"rollouts/{index:06d}/{rollout.name}", args.resource_limit, args.poll_seconds)
            add_file(archive, summary, f"rollouts/{index:06d}/{summary.name}", args.resource_limit, args.poll_seconds)
        optional = {
            "rollout-ledger.json": league_root / "learners" / CHAIN / "rollout-ledger.json",
            "league.json": league_root / "state/league.json",
            "full-matrix-latest.json": league_root / "monitoring/full-matrix/latest.json",
            "submission4-latest.json": league_root / "monitoring/submission4/latest.json",
        }
        for arcname, source in optional.items():
            if source.is_file():
                add_file(archive, source, f"metadata/{arcname}", args.resource_limit, args.poll_seconds)

    receipt = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "uploadedToKaggle": False,
        "chain": CHAIN,
        "generation": generation,
        "snapshotId": current.get("snapshotId"),
        "checkpoint": current.get("checkpoint"),
        "agentSource": str(agent_dir),
        "agentZip": {"path": str(agent_zip), "bytes": agent_zip.stat().st_size},
        "gamesArchive": {
            "path": str(games_tar),
            "bytes": games_tar.stat().st_size,
            "rolloutShards": len(rollout_pairs),
        },
        "resourceLimitPercent": args.resource_limit,
        "hashVerificationSkipped": True,
    }
    receipt_path = destination / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
