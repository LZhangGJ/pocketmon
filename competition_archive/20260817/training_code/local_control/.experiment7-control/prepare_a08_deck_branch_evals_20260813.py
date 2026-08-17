#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


CHAIN = "a08_dipplin_seaking"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(name: str, generation: int, args: argparse.Namespace) -> dict:
    branch = args.branch_root / name
    generation_root = branch / f"generation-{generation:04d}"
    checkpoint = generation_root / "checkpoint.pt"
    metrics = generation_root / "metrics.json"
    deck = args.branch_root / "decks" / f"{name}.csv"
    for required in (checkpoint, metrics, deck):
        if not required.is_file() or required.stat().st_size == 0:
            raise FileNotFoundError(required)

    candidate = args.output_root / f"{name}_g{generation:04d}"
    deployment = candidate / "deployment"
    portable = deployment / "universal_ppo.npz"
    package_root = deployment / "packages"
    agent_name = f"branch_{name}_g{generation:04d}"
    agent_dir = package_root / agent_name
    package_manifest = package_root / "packages.json"

    if not portable.is_file():
        deployment.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(args.python), "-s", str(args.exporter), "export",
                "--checkpoint", str(checkpoint), "--output", str(portable),
            ],
            check=True,
        )
    if not agent_dir.is_dir():
        shutil.copytree(args.template_agent, agent_dir)
    shutil.copy2(portable, agent_dir / "deck_identity_bc.npz")
    shutil.copy2(deck, agent_dir / "deck.csv")

    packages = {
        "schemaVersion": 1,
        "architecture": "experiment7_universal_deck8_autoregressive_stop",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "portable": {"path": str(portable)},
        "packages": [
            {
                "name": agent_name,
                "agentDir": str(agent_dir),
                "archetypeId": "A08",
                "archetypeLabel": name,
            }
        ],
    }
    write(package_manifest, packages)

    league = read(args.live_league)
    source_chain = copy.deepcopy(league["chains"][CHAIN])
    source_chain["deckName"] = name
    source_chain["archetypeLabel"] = name
    source_chain["deckPath"] = str(deck)
    source_chain["current"] = {
        "generation": generation,
        "checkpoint": str(checkpoint),
        "metrics": str(metrics),
        "snapshotId": f"{name}-g{generation:04d}",
        "packageManifest": str(package_manifest),
    }
    source_chain["history"] = []
    league["chains"] = {CHAIN: source_chain}
    league.pop("trainingControlUpdatedAt", None)
    league.pop("trainingControlSourceRoundId", None)
    write(candidate / "state" / "league.json", league)
    return {
        "branch": name,
        "generation": generation,
        "root": str(candidate),
        "checkpoint": str(checkpoint),
        "deck": str(deck),
        "packageManifest": str(package_manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--live-league", type=Path, required=True)
    parser.add_argument("--template-agent", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, required=True)
    parser.add_argument("--candidate", nargs=2, action="append", metavar=("NAME", "GEN"), required=True)
    args = parser.parse_args()
    receipt = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "candidates": [prepare(name, int(generation), args) for name, generation in args.candidate],
    }
    write(args.output_root / "candidates.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
