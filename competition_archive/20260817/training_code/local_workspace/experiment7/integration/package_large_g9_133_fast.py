import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.relative_to(root).parts:
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast immutable 133-deck packaging from one Large g9 template")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--template-agent", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name-prefix", default="large_g9_133")
    args = parser.parse_args()

    selected = json.loads(args.catalog.read_text(encoding="utf-8"))["selected"]
    if len(selected) != 133 or len({row["deckSha256"] for row in selected}) != 133:
        raise ValueError("catalog must contain 133 unique decks")
    required_names = {
        "main.py", "portable.py", "deck_identity_portable.py", "tokenizer.py",
        "features_vendor.py", "features.py", "deck_identity_bc.npz", "engine_catalog.json",
    }
    template_files = {
        path.name: path for path in args.template_agent.iterdir()
        if path.is_file() and path.name not in {"deck.csv", "receipt.json"}
    }
    missing = required_names - set(template_files)
    if missing:
        raise FileNotFoundError(f"template missing files: {sorted(missing)}")

    output = args.output_root.resolve()
    if output.name != "packages" or "large-g9-133-opponent-pool-20260814" not in output.parent.name:
        raise ValueError(f"refusing unexpected output target: {output}")
    build = output.with_name(f".{output.name}.build-{os.getpid()}")
    build.mkdir(parents=True, exist_ok=False)
    packages = []
    for row in selected:
        deck = Path(row["deckPath"])
        if not deck.is_file():
            raise FileNotFoundError(deck)
        name = f"{args.name_prefix}__{row['name']}"
        agent = build / name
        agent.mkdir()
        for filename, source in template_files.items():
            link_or_copy(source, agent / filename)
        shutil.copy2(deck, agent / "deck.csv")
        # Python sources are immutable hard links from the already compiled
        # and battle-tested template.  Recompiling 133 identical copies would
        # only add shared-filesystem traffic.
        receipt = {
            "schemaVersion": 1,
            "createdAt": utc_now(),
            "architecture": "experiment7_universal_deck8_autoregressive_stop",
            "name": name,
            "archetypeId": row["archetypeId"],
            "archetypeLabel": row["archetypeLabel"],
            "deckSha256": row["deckSha256"],
            "sourceDeck": str(deck.resolve()),
            "templateAgent": str(args.template_agent.resolve()),
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in sorted(agent.iterdir()) if path.is_file()
            },
        }
        atomic_json(agent / "receipt.json", receipt)
        packages.append({
            "name": name,
            "agentDir": str((output / name).resolve()),
            "deckSha256": row["deckSha256"],
            "archetypeId": row["archetypeId"],
            "archetypeLabel": row["archetypeLabel"],
            "directorySha256": directory_sha256(agent),
        })

    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "architecture": "experiment7_universal_deck8_autoregressive_stop",
        "catalog": {"path": str(args.catalog.resolve()), "sha256": sha256_file(args.catalog)},
        "templateAgent": str(args.template_agent.resolve()),
        "packages": packages,
        "optimization": "immutable common files hard-linked; copy fallback across filesystems",
    }
    atomic_json(build / "packages.json", payload)
    if output.exists():
        partial = output.with_name(f".{output.name}.partial-{int(datetime.now().timestamp())}")
        os.replace(output, partial)
    os.replace(build, output)
    print(json.dumps({"packages": len(packages), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
