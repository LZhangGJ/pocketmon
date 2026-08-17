#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import random
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FAILURES = {"crash", "timeout", "illegal"}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def paired(schedule: list[dict[str, Any]], learner: str, opponent: str, games: int, seed: int) -> None:
    if games % 2:
        raise ValueError("games per opponent must be even")
    for index in range(games // 2):
        schedule.append({"learner": learner, "opponent": opponent, "seed": seed + index, "learner_seat": 0})
        schedule.append({"learner": learner, "opponent": opponent, "seed": seed + index, "learner_seat": 1})


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate selected fixed agents against the complete frozen pool")
    parser.add_argument("--league-root", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--candidate", nargs=3, action="append", metavar=("NAME", "GENERATION", "AGENT_DIR"), required=True)
    parser.add_argument("--exclude-opponent", action="append", default=[])
    parser.add_argument("--games-per-opponent", type=int, default=4)
    parser.add_argument("--shards", type=int, default=30)
    parser.add_argument("--distributed-hosts", required=True)
    parser.add_argument("--max-shards-per-host", type=int, default=3)
    parser.add_argument("--remote-run-shard", default="/homes/lzhang/run_load_guarded_arena_shard.sh")
    args = parser.parse_args()

    league_root = args.league_root.resolve()
    league = load(league_root / "state/league.json")
    pool = load(Path(league["basePool"]["path"]))
    excluded = set(args.exclude_opponent)
    opponents = [
        row for row in pool.get("agents", [])
        if row.get("status", "accepted") == "accepted" and row["name"] not in excluded
    ]
    if not opponents:
        raise RuntimeError("no opponents selected")

    candidates = []
    for name, generation_text, directory_text in args.candidate:
        agent_dir = Path(directory_text).resolve()
        for required in (agent_dir / "main.py", agent_dir / "deck.csv"):
            if not required.is_file():
                raise FileNotFoundError(required)
        candidates.append(
            {
                "name": name,
                "generation": None if generation_text == "external" else int(generation_text),
                "agent_dir": str(agent_dir),
                "status": "accepted",
            }
        )
    total_per_candidate = len(opponents) * args.games_per_opponent
    eval_root = league_root / "monitoring" / "selected-best-vs-frozen-pool"
    rounds = eval_root / "rounds"
    rounds.mkdir(parents=True, exist_ok=True)
    lock = (eval_root / "eval.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"status": "busy", "latest": str(eval_root / "latest.json")}))
        return 0

    started = datetime.now(timezone.utc)
    candidate_tag = "-".join(
        f"{candidate['name']}-g{candidate['generation']}" if candidate["generation"] is not None else candidate["name"]
        for candidate in candidates
    )
    round_id = started.strftime("%Y%m%dT%H%M%SZ") + f"-{candidate_tag}-{args.games_per_opponent}g-per-opponent"
    staging = rounds / f".{round_id}.in-progress"
    final = rounds / round_id
    staging.mkdir()
    raw_dir = staging / "raw"
    log_dir = staging / "logs"
    raw_dir.mkdir()
    log_dir.mkdir()
    write(staging / "learners.json", {"schemaVersion": 1, "agents": candidates})
    write(staging / "opponents.json", {"schemaVersion": 1, "agents": opponents})

    schedule: list[dict[str, Any]] = []
    seed_base = 260_811_900
    for candidate_index, candidate in enumerate(candidates):
        for opponent_index, opponent in enumerate(opponents):
            paired(
                schedule,
                candidate["name"],
                opponent["name"],
                args.games_per_opponent,
                seed_base + candidate_index * 1_000_000 + opponent_index * 1_000,
            )
    random.Random(seed_base).shuffle(schedule)
    schedule_path = staging / "schedule.csv"
    with schedule_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("learner", "opponent", "seed", "learner_seat"))
        writer.writeheader()
        writer.writerows(schedule)

    requested = [host.strip() for host in args.distributed_hosts.split(",") if host.strip()]

    def probe(host: str) -> tuple[str, bool]:
        try:
            result = subprocess.run(
                [
                    "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                    f"lzhang@{host}",
                    f"test -x {shlex.quote(args.remote_run_shard)} && command -v bwrap >/dev/null",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=180,
            )
            return host, result.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return host, False

    with ThreadPoolExecutor(max_workers=len(requested)) as executor:
        available = [host for host, ok in executor.map(probe, requested) if ok]
    if not available:
        raise RuntimeError("no distributed hosts available")
    shard_count = min(args.shards, len(schedule), len(available) * args.max_shards_per_host)

    metadata = {
        "schemaVersion": 1,
        "status": "running",
        "roundId": round_id,
        "startedAt": started.isoformat(),
        "engineSeedControlled": False,
        "candidates": candidates,
        "opponents": [row["name"] for row in opponents],
        "excludedOpponents": sorted(excluded),
        "gamesPerOpponent": args.games_per_opponent,
        "gamesPerCandidate": total_per_candidate,
        "distributedHosts": available,
        "shards": shard_count,
        "maxShardsPerHost": args.max_shards_per_host,
    }
    write(staging / "metadata.json", metadata)

    def shard_args(index: int) -> list[str]:
        return [
            str(args.worktree.resolve()),
            str(args.python.resolve()),
            str(schedule_path),
            str(staging / "learners.json"),
            str(staging / "opponents.json"),
            str(Path(league["cgDir"])),
            str(raw_dir / f"results-shard-{index:03d}.csv"),
            str(index),
            str(shard_count),
            str(eval_root),
        ]

    processes = []
    for host_index, host in enumerate(available):
        indices = list(range(host_index, shard_count, len(available)))
        if not indices:
            continue
        commands = []
        for index in indices:
            command = [args.remote_run_shard, *shard_args(index)]
            commands.append(f"{shlex.join(command)} & pids=\"$pids $!\"")
        remote = "set -u; pids=\"\"; " + "; ".join(commands) + "; failed=0; for pid in $pids; do wait $pid || failed=1; done; exit $failed"
        log_handle = (log_dir / f"host-{host.replace('.', '-')}.log").open("w", encoding="utf-8")
        command = [
            "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=12",
            f"lzhang@{host}", remote,
        ]
        processes.append((host, subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True), log_handle))

    failures = []
    for host, process, handle in processes:
        code = process.wait()
        handle.close()
        if code:
            failures.append({"host": host, "returnCode": code})
    if failures:
        raise RuntimeError(f"distributed shard failures: {failures}")

    rows: list[dict[str, str]] = []
    for path in sorted(raw_dir.glob("results-shard-*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if len(rows) != len(schedule):
        raise RuntimeError(f"coverage mismatch: {len(rows)} != {len(schedule)}")

    report_rows = []
    for candidate in candidates:
        candidate_rows = [row for row in rows if row["learner"] == candidate["name"]]
        per_opponent = []
        for opponent in opponents:
            selected_rows = [row for row in candidate_rows if row["opponent"] == opponent["name"]]
            per_opponent.append({"agent": opponent["name"], **metric(selected_rows)})
        report_rows.append(
            {
                **candidate,
                "aggregate": metric(candidate_rows),
                "seat0": metric([row for row in candidate_rows if row["learner_seat"] == "0"]),
                "seat1": metric([row for row in candidate_rows if row["learner_seat"] == "1"]),
                "agents": per_opponent,
            }
        )
    completed = datetime.now(timezone.utc)
    report = {
        **metadata,
        "status": "complete",
        "completedAt": completed.isoformat(),
        "games": len(rows),
        "results": report_rows,
    }
    write(staging / "report.json", report)
    os.replace(staging, final)
    report["roundPath"] = str(final)
    latest_tmp = eval_root / f".latest.{os.getpid()}.tmp"
    write(latest_tmp, report)
    os.replace(latest_tmp, eval_root / "latest.json")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
