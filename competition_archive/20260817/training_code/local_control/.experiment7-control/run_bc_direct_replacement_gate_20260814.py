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


MAIN = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")
SCREENING = Path(
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/"
    "0812-d14-ram-npz-fast-20260813/replacement-screening"
)
WORKTREE = Path("/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0")
PYTHON = Path("/homes/lzhang/mypath/new/envs/trans/bin/python")
RUN_SHARD = Path("/homes/lzhang/run_load_guarded_arena_shard.sh")
HOSTS = (
    "10.113.13.53,10.113.13.54,10.113.13.57,10.113.13.63,10.113.13.64,"
    "10.113.13.67,10.113.13.68,10.113.13.69,10.113.13.71,10.113.13.72,"
    "10.113.13.73,10.113.13.74,10.113.13.75,10.113.13.77,10.113.13.78"
)
CURRENT_MANIFEST = (
    SCREENING
    / "current_bc-frozen40/monitoring/full-matrix/universal-bc-baseline/packages/packages.json"
)
CANDIDATE_MANIFESTS = {
    "standard_1m": (
        SCREENING
        / "standard_1m-frozen40/monitoring/full-matrix/universal-bc-baseline/packages/packages.json"
    ),
    "large_256x6": (
        SCREENING
        / "large_256x6-frozen40/monitoring/full-matrix/universal-bc-baseline/packages/packages.json"
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def paired_games(
    rows: list[dict[str, Any]], learner: str, opponent: str, games: int, seed_base: int
) -> None:
    if games <= 0 or games % 2:
        raise ValueError("games must be a positive even number")
    for pair_index in range(games // 2):
        seed = seed_base + pair_index
        rows.append({"learner": learner, "opponent": opponent, "seed": seed, "learner_seat": 0})
        rows.append({"learner": learner, "opponent": opponent, "seed": seed, "learner_seat": 1})


def renamed(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        "name": f"{prefix}__{row['name']}",
        "agent_dir": row["agentDir"],
        "status": "accepted",
        "directorySha256": row.get("directorySha256", ""),
        "deckSha256": row["deckSha256"],
        "archetypeId": row.get("archetypeId", ""),
        "archetypeLabel": row.get("archetypeLabel", ""),
    }


def reachable(host: str) -> tuple[str, bool]:
    try:
        completed = subprocess.run(
            [
                "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                f"lzhang@{host}",
                (
                    "bash --noprofile --norc -c '"
                    "test -x /homes/lzhang/run_load_guarded_arena_shard.sh "
                    "&& command -v bwrap >/dev/null "
                    "&& bwrap --unshare-net --ro-bind / / --dev /dev --proc /proc true"
                    "'"
                ),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=180,
        )
        return host, completed.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return host, False


def metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    wins = sum(row.get("result") == "win" for row in rows)
    losses = sum(row.get("result") == "loss" for row in rows)
    draws = sum(row.get("result") == "draw" for row in rows)
    failures = sum(row.get("result") in {"crash", "timeout", "illegal"} for row in rows)
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


def run_round(profile: str, round_index: int, games: int, hosts: list[str]) -> dict[str, Any]:
    root = SCREENING / "direct-new-old" / profile / f"round-{round_index}"
    summary_path = root / "summary.json"
    if summary_path.is_file() and load(summary_path).get("status") == "complete":
        return load(summary_path)
    root.mkdir(parents=True, exist_ok=True)
    lock_handle = (root / "round.lock").open("a+")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    current_rows = {
        row["deckSha256"]: row for row in load(CURRENT_MANIFEST).get("packages", [])
    }
    candidate_rows = {
        row["deckSha256"]: row
        for row in load(CANDIDATE_MANIFESTS[profile]).get("packages", [])
    }
    if set(current_rows) != set(candidate_rows) or len(current_rows) != 4:
        raise ValueError("direct gate requires the same four deck identities")

    learners = [renamed(candidate_rows[deck], f"candidate_{profile}") for deck in sorted(current_rows)]
    opponents = [renamed(current_rows[deck], "current_bc") for deck in sorted(current_rows)]
    atomic_json(root / "learners.json", {"schemaVersion": 1, "agents": learners})
    atomic_json(root / "opponents.json", {"schemaVersion": 1, "agents": opponents})

    schedule: list[dict[str, Any]] = []
    pairs: list[dict[str, str]] = []
    seed_base = 260_814_000 + round_index * 1_000_000
    for deck_index, deck in enumerate(sorted(current_rows)):
        learner = next(row["name"] for row in learners if row["deckSha256"] == deck)
        opponent = next(row["name"] for row in opponents if row["deckSha256"] == deck)
        paired_games(schedule, learner, opponent, games, seed_base + deck_index * 10_000)
        pairs.append({"deckSha256": deck, "learner": learner, "opponent": opponent})
    random.Random(seed_base).shuffle(schedule)
    with (root / "schedule.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("learner", "opponent", "seed", "learner_seat"))
        writer.writeheader()
        writer.writerows(schedule)

    available = [host for host in hosts if reachable(host)[1]]
    if not available:
        raise RuntimeError("no guarded Arena host reachable")
    shard_count = min(16, len(schedule), len(available))
    raw = root / "raw"
    logs = root / "logs"
    raw.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    league = load(MAIN / "state/league.json")

    processes: list[tuple[str, subprocess.Popen[str], Any]] = []
    for shard_index in range(shard_count):
        host = available[shard_index % len(available)]
        output = raw / f"results-shard-{shard_index:03d}.csv"
        if output.is_file():
            continue
        arguments = [
            str(WORKTREE), str(PYTHON), str(root / "schedule.csv"),
            str(root / "learners.json"), str(root / "opponents.json"),
            str(Path(league["cgDir"])), str(output), str(shard_index), str(shard_count), str(root),
        ]
        remote = shlex.join([str(RUN_SHARD), *arguments])
        log_handle = (logs / f"shard-{shard_index:03d}-{host.replace('.', '-')}.log").open(
            "a", encoding="utf-8"
        )
        process = subprocess.Popen(
            [
                "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=8",
                f"lzhang@{host}", remote,
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((host, process, log_handle))
    failures: list[dict[str, Any]] = []
    for host, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failures.append({"host": host, "returnCode": return_code})
    if failures:
        atomic_json(root / "FAILED.json", {"status": "failed", "at": utc_now(), "failures": failures})
        raise RuntimeError(f"Arena shard failures: {failures}")

    result_rows: list[dict[str, str]] = []
    for path in sorted(raw.glob("results-shard-*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            result_rows.extend(csv.DictReader(handle))
    by_deck: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        rows = [row for row in result_rows if row["learner"] == pair["learner"]]
        by_deck[pair["deckSha256"]] = {
            "archetype": candidate_rows[pair["deckSha256"]].get("archetypeLabel", ""),
            **metrics(rows),
            "seat0": metrics([row for row in rows if row.get("learner_seat") == "0"]),
            "seat1": metrics([row for row in rows if row.get("learner_seat") == "1"]),
        }
    summary = {
        "schemaVersion": 1,
        "status": "complete",
        "completedAt": utc_now(),
        "profile": profile,
        "round": round_index,
        "engine_seed_controlled": False,
        "gamesPerDeck": games,
        "hosts": available[:shard_count],
        "aggregate": metrics(result_rows),
        "byDeck": by_deck,
    }
    atomic_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired-seat same-deck candidate BC vs current BC gate")
    parser.add_argument("--profile", choices=tuple(CANDIDATE_MANIFESTS), required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--games-per-deck", type=int, default=40)
    parser.add_argument("--hosts", default=HOSTS)
    args = parser.parse_args()
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")
    hosts = [host.strip() for host in args.hosts.split(",") if host.strip()]
    summaries = [
        run_round(args.profile, round_index, args.games_per_deck, hosts)
        for round_index in range(1, args.rounds + 1)
    ]
    atomic_json(
        SCREENING / "direct-new-old" / args.profile / "complete.json",
        {"status": "complete", "profile": args.profile, "completedAt": utc_now(), "rounds": summaries},
    )


if __name__ == "__main__":
    main()
