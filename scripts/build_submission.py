from __future__ import annotations

import argparse
import json
import py_compile
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_deck(path: Path) -> list[int]:
    cards = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"Deck must contain exactly 60 cards, got {len(cards)}")
    return cards


def validate_cg(path: Path) -> None:
    required = (path / "api.py", path / "game.py", path / "sim.py")
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"Invalid cg directory; missing: {', '.join(missing)}")


def build(agent_dir: Path, cg_dir: Path, output: Path) -> Path:
    main_py = agent_dir / "main.py"
    deck_csv = agent_dir / "deck.csv"
    if not main_py.is_file():
        raise FileNotFoundError(main_py)
    if not deck_csv.is_file():
        raise FileNotFoundError(deck_csv)

    read_deck(deck_csv)
    validate_cg(cg_dir)
    py_compile.compile(str(main_py), doraise=True)

    manifest_path = agent_dir / "agent_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    is_rl_package = manifest.get("kind") == "our_rl_bc_specialist"
    if is_rl_package:
        required_payload = (agent_dir / "checkpoint.pt", agent_dir / "rl" / "__init__.py")
        missing_payload = [str(item) for item in required_payload if not item.is_file()]
        if missing_payload:
            raise FileNotFoundError(f"RL package is incomplete; missing: {', '.join(missing_payload)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        archive.add(main_py, arcname="main.py")
        archive.add(deck_csv, arcname="deck.csv")
        for filename in ("checkpoint.pt", "agent_manifest.json"):
            payload = agent_dir / filename
            if payload.is_file():
                archive.add(payload, arcname=filename)
        if (agent_dir / "rl").is_dir():
            archive.add(
                agent_dir / "rl",
                arcname="rl",
                filter=lambda info: None if "__pycache__" in info.name or info.name.endswith(".pyc") else info,
            )
        archive.add(cg_dir, arcname="cg", filter=lambda info: None if "__pycache__" in info.name else info)

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    required_members = {"main.py", "deck.csv", "cg/api.py", "cg/game.py", "cg/sim.py"}
    if is_rl_package:
        required_members.update({"checkpoint.pt", "agent_manifest.json", "rl/__init__.py", "rl/agent_adapter.py"})
    missing_members = sorted(required_members - names)
    if missing_members:
        raise RuntimeError(f"Archive validation failed; missing: {missing_members}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PTCG simulation submission archive")
    parser.add_argument("--agent", default="agents/lucario_rule", help="Directory containing main.py and deck.csv")
    parser.add_argument("--cg-dir", required=True, help="Official competition cg package directory")
    parser.add_argument("--output", default="dist/lucario_rule_submission.tar.gz")
    args = parser.parse_args()

    archive = build(
        (ROOT / args.agent).resolve() if not Path(args.agent).is_absolute() else Path(args.agent),
        (ROOT / args.cg_dir).resolve() if not Path(args.cg_dir).is_absolute() else Path(args.cg_dir),
        (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output),
    )
    print(f"archive={archive}")
    print(f"size_bytes={archive.stat().st_size}")


if __name__ == "__main__":
    main()
