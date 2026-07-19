from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def digest_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_agent(root: Path) -> None:
    required = [root / "main.py", root / "deck.csv"]
    missing = [p.name for p in required if not p.is_file()]
    if missing:
        raise ValueError(f"missing required files: {', '.join(missing)}")
    if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > 180 * 1024 * 1024:
        raise ValueError("agent exceeds the 180 MiB quarantine limit")
    subprocess.run(["python", "-m", "py_compile", str(root / "main.py")], check=True, timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download curated public notebooks into an immutable daily snapshot")
    parser.add_argument("--candidates", required=True, help="JSON list with name and Kaggle kernel source slug")
    parser.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--root", default="data/opponents/snapshots")
    args = parser.parse_args()
    candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    snapshot_dir = Path(args.root) / args.date
    if snapshot_dir.exists():
        raise FileExistsError(f"immutable snapshot already exists: {snapshot_dir}")
    snapshot_dir.mkdir(parents=True)
    manifest = {"date": args.date, "created_at": datetime.now(timezone.utc).isoformat(), "agents": []}
    for item in candidates:
        record = {"name": item["name"], "source": item["source"], "status": "rejected"}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run(["kaggle", "kernels", "pull", item["source"], "-p", tmp, "-m"], check=True, timeout=300)
                root = Path(tmp)
                validate_agent(root)
                sha = digest_tree(root)
                destination = snapshot_dir / f"{item['name']}--{sha[:12]}"
                shutil.copytree(root, destination)
                record.update({"status": "accepted", "sha256": sha, "path": str(destination)})
        except Exception as exc:
            record["reason"] = f"{type(exc).__name__}: {exc}"
        manifest["agents"].append(record)
    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
