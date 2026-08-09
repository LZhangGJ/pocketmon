from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import directory_sha256, sha256_file, stable_runtime_files, utc_now, write_json


def git_value(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the Lucario target and official engine receipts")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--cg-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    agent = args.agent_dir.resolve()
    cg = args.cg_dir.resolve()
    files = {}
    for path in stable_runtime_files(agent):
        files[path.relative_to(agent).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    engine_files = {}
    for path in stable_runtime_files(cg):
        if path.suffix in {".so", ".py", ".json"} or path.name.startswith("lib"):
            engine_files[path.relative_to(cg).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    try:
        import kaggle_environments  # type: ignore

        kaggle_version = getattr(kaggle_environments, "__version__", "unknown")
    except Exception as exc:
        kaggle_version = f"unavailable:{type(exc).__name__}"
    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "kaggleEnvironments": kaggle_version,
        "repository": {
            "path": str(repo),
            "commit": git_value(repo, "rev-parse", "HEAD"),
            "branch": git_value(repo, "branch", "--show-current"),
            "dirty": bool(git_value(repo, "status", "--porcelain")),
        },
        "targetAgent": {
            "path": str(agent),
            "directorySha256": directory_sha256(agent),
            "files": files,
        },
        "engine": {
            "path": str(cg),
            "directorySha256": directory_sha256(cg),
            "files": engine_files,
            "seedControlled": False,
        },
    }
    write_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
