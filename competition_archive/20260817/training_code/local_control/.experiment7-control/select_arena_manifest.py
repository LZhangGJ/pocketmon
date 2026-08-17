from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and normalize a frozen Arena manifest")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-kind", choices=("packages", "agents"), required=True)
    parser.add_argument("--selected-names", nargs="+", required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    receipt = args.receipt.resolve()
    if output.exists() or receipt.exists():
        raise FileExistsError(f"refusing to overwrite manifest artifacts: {output} / {receipt}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    collection = "packages" if isinstance(payload.get("packages"), list) else "agents"
    rows = payload.get(collection)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"source has no packages or agents: {source}")

    by_name: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("name", ""))
        if not name or name in by_name:
            raise ValueError(f"invalid or duplicate source name: {name!r}")
        by_name[name] = row
    selected_names = list(dict.fromkeys(args.selected_names))
    missing = sorted(set(selected_names) - set(by_name))
    if missing:
        raise ValueError(f"selected names are missing from source: {missing}")

    selected = []
    for name in selected_names:
        source_row = by_name[name]
        agent_dir = first(source_row, "agentDir", "agent_dir", "path")
        directory_hash = first(source_row, "directorySha256", "directory_sha256")
        if not agent_dir or not isinstance(directory_hash, str) or len(directory_hash) != 64:
            raise ValueError(f"source row lacks frozen path/hash: {name}")
        if args.output_kind == "packages":
            row = {
                "name": name,
                "agentDir": str(agent_dir),
                "directorySha256": directory_hash.lower(),
                "status": "accepted",
            }
        else:
            row = {
                "name": name,
                "agent_dir": str(agent_dir),
                "directory_sha256": directory_hash.lower(),
                "status": "accepted",
            }
        for key in ("archetype", "deckCanonicalSha256", "deck_canonical_sha256"):
            if key in source_row:
                row[key] = source_row[key]
        selected.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump({"schemaVersion": 1, args.output_kind: selected}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    audit = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(source), "sha256": sha256_file(source), "collection": collection},
        "output": {"path": str(output), "sha256": sha256_file(output), "kind": args.output_kind},
        "selectedNames": selected_names,
        "selectedCount": len(selected),
        "sourceModified": False,
    }
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
