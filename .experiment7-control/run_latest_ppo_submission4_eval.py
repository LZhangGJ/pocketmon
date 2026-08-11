from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import random
import shutil
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUBMISSION4 = "team_submission_4_portable_bc"
FAILURES = {"crash", "timeout", "illegal"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write_json(temporary, payload)
    os.replace(temporary, path)


def metric(rows: list[dict[str, str]]) -> dict[str, Any]:
    wins = sum(row.get("result") == "win" for row in rows)
    losses = sum(row.get("result") == "loss" for row in rows)
    draws = sum(row.get("result") == "draw" for row in rows)
    failures = sum(row.get("result") in FAILURES for row in rows)
    completed = wins + losses + draws
    return {
        "games": len(rows),
        "completed": completed,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "failures": failures,
        "scoreRate": (wins + 0.5 * draws) / completed if completed else None,
    }


def read_result_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def latest_complete(eval_root: Path) -> dict[str, Any] | None:
    path = eval_root / "latest.json"
    return load_json(path) if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the latest PPO deployments against frozen submission4"
    )
    parser.add_argument("--league-root", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--run-shard", type=Path, required=True)
    parser.add_argument("--summarizer", type=Path, required=True)
    parser.add_argument("--games-per-agent", type=int, default=20)
    parser.add_argument("--shards", type=int, default=12)
    args = parser.parse_args()

    if args.games_per_agent < 2 or args.games_per_agent % 2:
        raise ValueError("games-per-agent must be a positive even number")
    league_root = args.league_root.resolve()
    eval_root = league_root / "monitoring" / "submission4"
    rounds_root = eval_root / "rounds"
    rounds_root.mkdir(parents=True, exist_ok=True)
    lock_path = eval_root / "eval.lock"
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        payload = latest_complete(eval_root) or {"status": "busy", "message": "no completed round yet"}
        payload = {**payload, "busy": True}
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    started = utc_now()
    league = load_json(league_root / "state" / "league.json")
    pool_path = Path(league.get("poolPath", league_root / "state" / "opponent-pool-live.json"))
    pool = load_json(pool_path)
    opponent = next(
        (row for row in pool.get("agents", []) if row.get("name") == SUBMISSION4), None
    )
    if not opponent or opponent.get("status", "accepted") != "accepted":
        raise RuntimeError(f"accepted opponent is missing from pool: {SUBMISSION4}")

    chains: dict[str, dict[str, Any]] = league["chains"]
    selected: list[dict[str, Any]] = []
    packages: list[dict[str, Any]] = []
    for chain_name, chain in chains.items():
        current = chain["current"]
        package_manifest = load_json(Path(current["packageManifest"]))
        package_rows = package_manifest.get("packages", [])
        if len(package_rows) != 1:
            raise ValueError(f"expected one package for {chain_name}, got {len(package_rows)}")
        package = package_rows[0]
        packages.append(package)
        selected.append(
            {
                "chain": chain_name,
                "generation": int(current["generation"]),
                "snapshotId": current["snapshotId"],
                "checkpointSha256": current["sha256"],
                "packageManifest": current["packageManifest"],
                "learner": package["name"],
                "agentDir": package["agentDir"],
                "directorySha256": package["directorySha256"],
            }
        )
    if not selected:
        raise ValueError("league contains no deployed PPO chains")

    generation_tag = "-".join(f"{row['chain']}-g{row['generation']:06d}" for row in selected)
    round_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}-{generation_tag}"
    staging = rounds_root / f".{round_id}.in-progress"
    final = rounds_root / round_id
    if staging.exists() or final.exists():
        raise FileExistsError(f"round already exists: {round_id}")
    staging.mkdir()

    metadata = {
        "schemaVersion": 1,
        "status": "running",
        "roundId": round_id,
        "startedAt": started.isoformat(),
        "gamesPerAgent": args.games_per_agent,
        "shards": args.shards,
        "opponent": SUBMISSION4,
        "poolPath": str(pool_path),
        "selected": selected,
    }
    write_json(staging / "metadata.json", metadata)
    write_json(staging / "packages.json", {"schemaVersion": 1, "packages": packages})
    learners = {
        "schemaVersion": 1,
        "agents": [
            {
                "name": row["name"],
                "agent_dir": row["agentDir"],
                "directorySha256": row["directorySha256"],
                "deckSha256": row["deckSha256"],
                "archetypeId": row.get("archetypeId"),
                "archetypeLabel": row.get("archetypeLabel"),
                "status": "accepted",
            }
            for row in packages
        ],
    }
    write_json(staging / "learners.json", learners)
    write_json(
        staging / "opponents.json",
        {
            **{key: value for key, value in pool.items() if key != "agents"},
            "agents": [opponent],
        },
    )

    seed_base = int(started.timestamp()) % 1_000_000_000
    schedule: list[dict[str, Any]] = []
    for learner_index, row in enumerate(selected):
        for pair_index in range(args.games_per_agent // 2):
            seed = seed_base + learner_index * 1_000_000 + pair_index
            schedule.append(
                {"learner": row["learner"], "opponent": SUBMISSION4, "seed": seed, "learner_seat": 0}
            )
            schedule.append(
                {"learner": row["learner"], "opponent": SUBMISSION4, "seed": seed, "learner_seat": 1}
            )
    random.Random(seed_base).shuffle(schedule)
    schedule_path = staging / "schedule.csv"
    with schedule_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("learner", "opponent", "seed", "learner_seat"))
        writer.writeheader()
        writer.writerows(schedule)

    raw_dir = staging / "raw"
    log_dir = staging / "logs"
    raw_dir.mkdir()
    log_dir.mkdir()
    processes: list[tuple[int, subprocess.Popen[str], Any]] = []
    shard_count = min(args.shards, len(schedule))
    guarded_runner = Path("/homes/lzhang/run_load_guarded_arena_shard.sh")
    shard_runner = guarded_runner if guarded_runner.is_file() else args.run_shard.resolve()
    for shard_index in range(shard_count):
        log_handle = (log_dir / f"shard-{shard_index:02d}.log").open("w", encoding="utf-8")
        command = [
            "bash",
            str(shard_runner),
            str(args.worktree.resolve()),
            str(args.python.resolve()),
            str(schedule_path),
            str(staging / "learners.json"),
            str(staging / "opponents.json"),
            str(Path(league["cgDir"]).resolve()),
            str(raw_dir / f"results-shard-{shard_index:02d}.csv"),
            str(shard_index),
            str(shard_count),
            str(eval_root),
        ]
        processes.append(
            (
                shard_index,
                subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True),
                log_handle,
            )
        )
    failures: list[dict[str, Any]] = []
    for shard_index, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failures.append({"shard": shard_index, "returnCode": return_code})
    if failures:
        raise RuntimeError(f"Arena shard failures: {failures}")

    result_paths = sorted(raw_dir.glob("results-shard-*.csv"))
    summary_dir = staging / "summary"
    subprocess.run(
        [
            str(args.python.resolve()),
            str(args.summarizer.resolve()),
            "--schedule",
            str(schedule_path),
            "--results",
            *[str(path) for path in result_paths],
            "--learners",
            str(staging / "learners.json"),
            "--opponents",
            str(staging / "opponents.json"),
            "--output-dir",
            str(summary_dir),
        ],
        check=True,
    )
    round_rows = read_result_rows(result_paths)
    chain_by_learner = {row["learner"]: row["chain"] for row in selected}
    current_metrics = {
        chain: metric([row for row in round_rows if chain_by_learner[row["learner"]] == chain])
        for chain in chain_by_learner.values()
    }

    metadata.update({"status": "complete", "completedAt": utc_now().isoformat()})
    write_json(staging / "metadata.json", metadata)
    os.replace(staging, final)

    cumulative_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    complete_rounds = 0
    for round_dir in sorted(path for path in rounds_root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        round_metadata_path = round_dir / "metadata.json"
        if not round_metadata_path.exists():
            continue
        round_metadata = load_json(round_metadata_path)
        if round_metadata.get("status") != "complete":
            continue
        mapping = {row["learner"]: row["chain"] for row in round_metadata["selected"]}
        rows = read_result_rows(sorted((round_dir / "raw").glob("results-shard-*.csv")))
        for row in rows:
            if row["learner"] in mapping:
                cumulative_rows[mapping[row["learner"]]].append(row)
        complete_rounds += 1

    previous = latest_complete(eval_root)
    payload = {
        "schemaVersion": 1,
        "status": "complete",
        "busy": False,
        "updatedAt": utc_now().isoformat(),
        "roundId": round_id,
        "roundPath": str(final),
        "opponent": SUBMISSION4,
        "gamesPerAgent": args.games_per_agent,
        "selected": selected,
        "current": current_metrics,
        "cumulative": {chain: metric(rows) for chain, rows in cumulative_rows.items()},
        "completedRounds": complete_rounds,
        "previousRound": {
            "roundId": previous.get("roundId"),
            "current": previous.get("current"),
        }
        if previous
        else None,
    }
    atomic_write_json(eval_root / "latest.json", payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise
