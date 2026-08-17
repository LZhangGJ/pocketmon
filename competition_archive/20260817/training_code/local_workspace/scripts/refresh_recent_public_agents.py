from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def as_dict(value: object) -> dict:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return dict(vars(value))


def digest_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def find_agent_root(root: Path) -> Path:
    if (root / "main.py").is_file() and (root / "deck.csv").is_file():
        return root
    main_roots = {path.parent for path in root.rglob("main.py")}
    deck_roots = {path.parent for path in root.rglob("deck.csv")}
    candidates = sorted(main_roots & deck_roots)
    if not candidates:
        for archive in root.rglob("*.tar.gz"):
            unpacked = root / ("_unpacked_" + archive.name.removesuffix(".tar.gz"))
            unpacked.mkdir()
            with tarfile.open(archive) as handle:
                handle.extractall(unpacked, filter="data")
        main_roots = {path.parent for path in root.rglob("main.py")}
        deck_roots = {path.parent for path in root.rglob("deck.csv")}
        candidates = sorted(main_roots & deck_roots)
    if not candidates:
        raise ValueError(f"expected one main.py/deck.csv directory, found {len(candidates)}")
    # Notebook outputs often contain both the submission root and a nested copy.
    # The shallowest complete directory is the intended materialized agent.
    return min(candidates, key=lambda path: (len(path.relative_to(root).parts), str(path)))


def validate_agent(root: Path, python: str) -> None:
    total = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if total > 180 * 1024 * 1024:
        raise ValueError("agent exceeds 180 MiB quarantine limit")
    subprocess.run([python, "-m", "py_compile", str(root / "main.py")], check=True, timeout=30)


def list_all(api: KaggleApi, competition: str, sort_by: str) -> list[dict]:
    rows: list[dict] = []
    for page in range(1, 51):
        batch = api.kernels_list(
            competition=competition, sort_by=sort_by, page=page, page_size=200
        )
        rows.extend(as_dict(item) for item in batch)
        if len(batch) < 200:
            break
    return rows


def discover(api: KaggleApi, competition: str, days: int, limit: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    recent: dict[str, dict] = {}
    for row in list_all(api, competition, "dateCreated"):
        raw_time = row.get("lastRunTime")
        if not raw_time:
            continue
        run_time = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
        if run_time >= cutoff:
            recent[str(row["ref"])] = row

    ranked: list[dict] = []
    for public_rank, row in enumerate(list_all(api, competition, "scoreDescending"), 1):
        ref = str(row["ref"])
        if ref in recent:
            item = dict(recent[ref])
            item["overall_public_score_rank"] = public_rank
            item["recent_public_score_rank"] = len(ranked) + 1
            ranked.append(item)
            if len(ranked) == limit:
                break
    return ranked


def update(args: argparse.Namespace) -> Path:
    root = Path(args.root).resolve()
    snapshots = root / "snapshots"
    admin = root / "_pool_admin"
    snapshots.mkdir(parents=True, exist_ok=True)
    admin.mkdir(parents=True, exist_ok=True)
    lock_handle = (admin / "refresh.lock").open("w")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    api = KaggleApi()
    api.authenticate()
    selected = discover(api, args.competition, args.days, args.limit)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    staging = snapshots / f".{stamp}.staging"
    final = snapshots / stamp
    staging.mkdir()
    records: list[dict] = []

    for item in selected:
        ref = str(item["ref"])
        record = dict(item)
        record["status"] = "rejected"
        try:
            with tempfile.TemporaryDirectory(dir=admin) as tmp:
                subprocess.run(
                    [args.kaggle, "kernels", "output", ref, "-p", tmp, "--force"],
                    check=True,
                    timeout=args.download_timeout,
                )
                agent_root = find_agent_root(Path(tmp))
                validate_agent(agent_root, args.python)
                sha = digest_tree(agent_root)
                name = ref.replace("/", "__") + "--" + sha[:12]
                shutil.copytree(agent_root, staging / name)
                record.update(status="accepted", sha256=sha, directory=name)
        except Exception as exc:
            record["reason"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    manifest = {
        "competition": args.competition,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recency_days": args.days,
        "limit": args.limit,
        "ranking": "Kaggle kernels scoreDescending; numeric score is not exposed by the API",
        "selected_count": len(selected),
        "accepted_count": sum(row["status"] == "accepted" for row in records),
        "agents": records,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (staging / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in records for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    staging.rename(final)

    link_tmp = root / ".current.tmp"
    if link_tmp.is_symlink() or link_tmp.exists():
        link_tmp.unlink()
    link_tmp.symlink_to(final.relative_to(root), target_is_directory=True)
    os.replace(link_tmp, root / "current")
    (admin / "last_success.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh recent high-public-score Kaggle agents")
    parser.add_argument("--root", required=True)
    parser.add_argument("--competition", default="pokemon-tcg-ai-battle")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--kaggle", default="kaggle")
    parser.add_argument("--download-timeout", type=int, default=600)
    args = parser.parse_args()
    print(update(args))


if __name__ == "__main__":
    main()
