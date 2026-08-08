from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_deck(path: Path) -> list[int]:
    deck = [
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(deck) != 60:
        raise ValueError(f"deck must contain exactly 60 cards, got {len(deck)}")
    return deck


def materialize(
    checkpoint: Path,
    deck_path: Path,
    output: Path,
    name: str,
) -> dict[str, object]:
    checkpoint = checkpoint.resolve(strict=True)
    deck_path = deck_path.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing package: {output}")
    deck = read_deck(deck_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
    )
    try:
        shutil.copy2(
            ROOT / "agents" / "rl_bc_temporal_specialist" / "main.py",
            staging / "main.py",
        )
        shutil.copytree(
            ROOT / "rl",
            staging / "rl",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(checkpoint, staging / "checkpoint.pt")
        (staging / "deck.csv").write_text(
            "".join(f"{card_id}\n" for card_id in deck),
            encoding="utf-8",
        )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "name": name,
            "kind": "our_rl_bc_temporal_specialist",
            "checkpoint_sha256": sha256(staging / "checkpoint.pt"),
            "deck_sha256": sha256(staging / "deck.csv"),
            "main_sha256": sha256(staging / "main.py"),
            "card_count": len(deck),
            "confidence_threshold": 0.0,
            "history_length": 8,
        }
        (staging / "agent_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an immutable RL-BC-004 temporal specialist package"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                args.checkpoint,
                args.deck,
                args.output,
                args.name,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
