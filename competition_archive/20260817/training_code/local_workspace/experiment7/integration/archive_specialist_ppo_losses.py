from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CHAINS = (
    "a02_grim_g247",
    "a02_grim_g247_pokegear",
    "a08_rabsca",
    "a08_maxbelt",
    "lucario_gold_exact",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        with gzip.open(name, "wt", encoding="utf-8", compresslevel=6) as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load_summary(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def source_candidates(source_dir: Path, generations: set[int]) -> list[tuple[Path, dict[str, Any]]]:
    result: list[tuple[Path, dict[str, Any]]] = []
    for summary_path in source_dir.glob("*.jsonl.gz.summary.json"):
        summary = load_summary(summary_path)
        if not summary:
            continue
        generation = int(summary.get("behaviorGeneration", -1))
        if generation not in generations or int(summary.get("losses", 0)) <= 0:
            continue
        source = summary_path.with_suffix("").with_suffix("")
        if source.is_file():
            result.append((source, summary))
    result.sort(key=lambda item: (str(item[1].get("createdAt", "")), item[0].name), reverse=True)
    return result


def loss_episodes(source: Path, chain: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if bool(row.get("self_play", False)):
                continue
            groups[str(row["episode_id"])].append(row)
    episodes: list[dict[str, Any]] = []
    for episode_id, rows in groups.items():
        if not rows or not any(float(row.get("outcome", 0.0)) < 0 for row in rows):
            continue
        first = rows[0]
        episodes.append(
            {
                "schemaVersion": 1,
                "chain": chain,
                "episodeId": episode_id,
                "behaviorGeneration": int(first["behavior_generation"]),
                "behaviorSnapshotId": str(first["behavior_snapshot_id"]),
                "behaviorCheckpointSha256": str(first["behavior_checkpoint_sha256"]),
                "opponent": str(first.get("opponent", "unknown")),
                "learnerSeat": int(first["player"]),
                "decisionCount": len(rows),
                "sourceShard": source.name,
                "rows": rows,
            }
        )
    return episodes


def archive_chain(source_root: Path, archive_root: Path, chain: str, keep_generations: int, max_episodes: int) -> dict[str, Any]:
    source_dir = source_root / chain
    summaries = [load_summary(path) for path in source_dir.glob("*.jsonl.gz.summary.json")]
    # A chain can be deliberately reinitialized from a new BC checkpoint and
    # restart at a smaller generation number (for example Lucario g113 -> g1).
    # Recency must therefore come from immutable shard creation timestamps,
    # never from the numeric generation alone.
    generation_recency: dict[int, str] = {}
    for row in summaries:
        if not row or "behaviorGeneration" not in row:
            continue
        generation = int(row["behaviorGeneration"])
        created_at = str(row.get("createdAt", ""))
        generation_recency[generation] = max(
            created_at, generation_recency.get(generation, "")
        )
    generations = [
        generation
        for generation, _ in sorted(
            generation_recency.items(), key=lambda item: (item[1], item[0]), reverse=True
        )[:keep_generations]
    ]
    wanted = set(generations)
    candidates = source_candidates(source_dir, wanted)
    collected: list[dict[str, Any]] = []
    source_receipts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source, summary in candidates:
        added = 0
        for episode in loss_episodes(source, chain):
            key = episode["episodeId"]
            if key in seen:
                continue
            seen.add(key)
            collected.append(episode)
            added += 1
            if len(collected) >= max_episodes:
                break
        source_receipts.append(
            {
                "path": str(source.resolve()),
                "sha256": str((summary.get("output") or {}).get("sha256") or sha256_file(source)),
                "behaviorGeneration": int(summary["behaviorGeneration"]),
                "createdAt": summary.get("createdAt"),
                "archivedLossEpisodes": added,
            }
        )
        if len(collected) >= max_episodes:
            break

    by_generation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for episode in collected:
        by_generation[int(episode["behaviorGeneration"])].append(episode)
    chain_root = archive_root / chain
    outputs = []
    for generation in generations:
        rows = by_generation.get(generation, [])
        output = chain_root / f"generation-{generation:06d}.losses.jsonl.gz"
        atomic_gzip_jsonl(output, rows)
        outputs.append(
            {
                "generation": generation,
                "episodes": len(rows),
                "decisions": sum(int(row["decisionCount"]) for row in rows),
                "path": str(output.resolve()),
                "sha256": sha256_file(output),
            }
        )
    for stale in chain_root.glob("generation-*.losses.jsonl.gz"):
        if stale not in {Path(row["path"]) for row in outputs}:
            stale.unlink()
    receipt = {
        "schemaVersion": 1,
        "updatedAt": utc_now(),
        "chain": chain,
        "policy": {
            "externalLossesOnly": True,
            "selfPlayExcluded": True,
            "keepNewestGenerations": keep_generations,
            "maxEpisodesPerChain": max_episodes,
            "newestFirst": True,
        },
        "retainedGenerations": generations,
        "retainedEpisodes": len(collected),
        "outputs": outputs,
        "sources": source_receipts,
    }
    atomic_json(chain_root / "latest.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--chains", nargs="+", default=list(DEFAULT_CHAINS))
    parser.add_argument("--keep-generations", type=int, default=3)
    parser.add_argument("--max-episodes-per-chain", type=int, default=100)
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.keep_generations <= 0 or args.max_episodes_per_chain <= 0:
        raise ValueError("retention limits must be positive")
    if args.summary_only:
        receipts = []
        for chain in args.chains:
            receipt_path = args.archive_root.resolve() / chain / "latest.json"
            receipt = load_summary(receipt_path)
            if not receipt:
                raise FileNotFoundError(receipt_path)
            receipts.append(receipt)
        latest = {
            "schemaVersion": 1,
            "updatedAt": utc_now(),
            "chains": receipts,
            "totalEpisodes": sum(int(row["retainedEpisodes"]) for row in receipts),
        }
        atomic_json(args.archive_root.resolve() / "latest.json", latest)
        print(json.dumps(latest, sort_keys=True))
        return
    receipts = [
        archive_chain(
            args.source_root.resolve(),
            args.archive_root.resolve(),
            chain,
            args.keep_generations,
            args.max_episodes_per_chain,
        )
        for chain in args.chains
    ]
    latest = {
        "schemaVersion": 1,
        "updatedAt": utc_now(),
        "chains": receipts,
        "totalEpisodes": sum(int(row["retainedEpisodes"]) for row in receipts),
    }
    atomic_json(args.archive_root.resolve() / "latest.json", latest)
    print(json.dumps(latest, sort_keys=True))


if __name__ == "__main__":
    main()
