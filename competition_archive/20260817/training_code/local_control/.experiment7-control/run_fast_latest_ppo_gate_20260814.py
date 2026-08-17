from __future__ import annotations

import csv
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEAGUE_ROOT = Path(
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811"
)
WORKTREE = Path("/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0")
PYTHON = Path("/homes/lzhang/mypath/new/envs/trans/bin/python3.11")
EVAL_ROOT = LEAGUE_ROOT / "monitoring" / "fast-latest-gate"
PRIORITY_OPPONENTS = (
    "notebook_crustle_wall",
    "public_alakazam_search_v9",
    "team_grim_model_a",
    "public_archaludon_meta",
    "team_submission_3_boss_tactical",
    "public_lucario_search",
    "notebook_dragapult_rules",
    "champion_a08_dipplin_seaking_g000077",
)
FAILURES = {"crash", "timeout", "illegal"}


def now() -> datetime:
    return datetime.now(timezone.utc)


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def package_agent(package: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": package["name"],
        "agent_dir": package["agentDir"],
        "status": "accepted",
        "directorySha256": package.get("directorySha256", ""),
        "deckSha256": package.get("deckSha256", ""),
        "archetypeId": package.get("archetypeId", ""),
        "archetypeLabel": package.get("archetypeLabel", ""),
    }


def paired_games(
    schedule: list[dict[str, Any]], learner: str, opponent: str, games: int, seed_base: int
) -> None:
    if games <= 0 or games % 2:
        raise ValueError("games must be a positive even number")
    for index in range(games // 2):
        seed = seed_base + index
        schedule.append({"learner": learner, "opponent": opponent, "seed": seed, "learner_seat": 0})
        schedule.append({"learner": learner, "opponent": opponent, "seed": seed, "learner_seat": 1})


def metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
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
        "decisiveWinRate": wins / (wins + losses) if wins + losses else None,
    }


def select_frozen(base_pool: dict[str, Any]) -> list[dict[str, Any]]:
    accepted = [row for row in base_pool.get("agents", []) if row.get("status", "accepted") == "accepted"]
    by_name = {row["name"]: row for row in accepted}
    selected = [by_name[name] for name in PRIORITY_OPPONENTS if name in by_name]
    seen = {row["name"] for row in selected}
    for row in accepted:
        if len(selected) >= len(PRIORITY_OPPONENTS):
            break
        if row["name"] not in seen:
            selected.append(row)
            seen.add(row["name"])
    if not selected:
        raise RuntimeError("base pool has no accepted opponents")
    return selected


def main() -> None:
    started = now()
    league = load(LEAGUE_ROOT / "state" / "league.json")
    base_pool = load(Path(league["basePool"]["path"]))
    frozen = select_frozen(base_pool)

    bc_manifest = load(
        LEAGUE_ROOT
        / "monitoring"
        / "full-matrix"
        / "universal-bc-baseline"
        / "packages"
        / "packages.json"
    )
    prior_bc = {
        row.get("deckSha256", ""): package_agent(row)
        for row in bc_manifest.get("packages", [])
    }
    latest_full = load(LEAGUE_ROOT / "monitoring" / "full-matrix" / "latest.json")
    prior_learners = load(Path(latest_full["roundPath"]) / "learners.json")
    prior_bc.update(
        {
            row.get("deckSha256", ""): row
            for row in prior_learners.get("agents", [])
            if str(row.get("name", "")).startswith("universal_bc_baseline")
        }
    )

    selected: list[dict[str, Any]] = []
    learners: list[dict[str, Any]] = []
    opponents = list(frozen)
    opponent_names = {row["name"] for row in opponents}
    for chain_name, chain in sorted(league["chains"].items()):
        current = chain["current"]
        packages = load(Path(current["packageManifest"])).get("packages", [])
        if len(packages) != 1:
            raise RuntimeError(f"expected one current package for {chain_name}")
        package = packages[0]
        learners.append(package_agent(package))
        deck_sha = chain["deckSha256"]
        bc = prior_bc.get(deck_sha)
        if bc is None:
            raise RuntimeError(f"missing same-deck BC baseline for {chain_name}")
        if bc["name"] not in opponent_names:
            opponents.append(bc)
            opponent_names.add(bc["name"])
        selected.append(
            {
                "chain": chain_name,
                "generation": int(current["generation"]),
                "snapshotId": current["snapshotId"],
                "learner": package["name"],
                "bcOpponent": bc["name"],
                "deckSha256": deck_sha,
            }
        )

    generation_tag = "-".join(f"{row['chain']}-g{row['generation']:06d}" for row in selected)
    round_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}-{generation_tag}"
    round_root = EVAL_ROOT / "rounds" / round_id
    raw_root = round_root / "raw"
    logs_root = round_root / "logs"
    raw_root.mkdir(parents=True)
    logs_root.mkdir()
    write(round_root / "learners.json", {"schemaVersion": 1, "agents": learners})
    write(round_root / "opponents.json", {"schemaVersion": 1, "agents": opponents})

    schedule: list[dict[str, Any]] = []
    seed_base = 260_814_100
    for chain_index, row in enumerate(selected):
        for opponent_index, opponent in enumerate(frozen):
            paired_games(
                schedule,
                row["learner"],
                opponent["name"],
                4,
                seed_base + chain_index * 1_000_000 + opponent_index * 1_000,
            )
        paired_games(
            schedule,
            row["learner"],
            row["bcOpponent"],
            20,
            seed_base + 500_000_000 + chain_index * 10_000,
        )
    random.Random(seed_base).shuffle(schedule)
    schedule_path = round_root / "schedule.csv"
    with schedule_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("learner", "opponent", "seed", "learner_seat"))
        writer.writeheader()
        writer.writerows(schedule)

    shard_count = min(16, len(schedule))
    metadata = {
        "schemaVersion": 1,
        "status": "running",
        "roundId": round_id,
        "startedAt": started.isoformat(),
        "host": os.uname().nodename,
        "selected": selected,
        "frozenAgents": [row["name"] for row in frozen],
        "games": len(schedule),
        "shards": shard_count,
        "gamesPerFrozen": 4,
        "gamesVsSameDeckBc": 20,
        "maxDecisions": 400,
        "timeoutSeconds": 90,
        "engineSeedControlled": False,
    }
    write(round_root / "metadata.json", metadata)

    processes: list[tuple[int, subprocess.Popen[str], Any]] = []
    for shard_index in range(shard_count):
        output = raw_root / f"results-shard-{shard_index:03d}.csv"
        log_handle = (logs_root / f"shard-{shard_index:03d}.log").open("w", encoding="utf-8")
        command = [
            "bwrap",
            "--unshare-net",
            "--ro-bind", "/", "/",
            "--dev-bind", "/dev", "/dev",
            "--proc", "/proc",
            "--tmpfs", "/tmp",
            "--bind", str(EVAL_ROOT), str(EVAL_ROOT),
            "--chdir", str(WORKTREE),
            "--setenv", "OPENBLAS_NUM_THREADS", "1",
            "--setenv", "OMP_NUM_THREADS", "1",
            "--setenv", "MKL_NUM_THREADS", "1",
            "--setenv", "LD_LIBRARY_PATH", "/homes/lzhang/mypath/new/lib:/homes/lzhang/cuda12.9/cuda-12.9/lib64",
            str(PYTHON), str(WORKTREE / "scripts" / "run_league_schedule.py"),
            "--schedule", str(schedule_path),
            "--learners", str(round_root / "learners.json"),
            "--opponents", str(round_root / "opponents.json"),
            "--cg-dir", str(Path(league["cgDir"])),
            "--output", str(output),
            "--shard-index", str(shard_index),
            "--shard-count", str(shard_count),
            "--max-decisions", "400",
            "--timeout-seconds", "90",
        ]
        processes.append(
            (shard_index, subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True), log_handle)
        )

    failed_shards: list[dict[str, int]] = []
    for shard_index, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failed_shards.append({"shard": shard_index, "returnCode": return_code})

    rows: list[dict[str, str]] = []
    for path in sorted(raw_root.glob("results-shard-*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))

    frozen_names = {row["name"] for row in frozen}
    chains: dict[str, Any] = {}
    for selected_row in selected:
        learner_rows = [row for row in rows if row.get("learner") == selected_row["learner"]]
        frozen_rows = [row for row in learner_rows if row.get("opponent") in frozen_names]
        bc_rows = [row for row in learner_rows if row.get("opponent") == selected_row["bcOpponent"]]
        chains[selected_row["chain"]] = {
            "generation": selected_row["generation"],
            "snapshotId": selected_row["snapshotId"],
            "frozenAggregate": metrics(frozen_rows),
            "directVsSameDeckBc": metrics(bc_rows),
            "seat0": metrics([row for row in frozen_rows if row.get("learner_seat") == "0"]),
            "seat1": metrics([row for row in frozen_rows if row.get("learner_seat") == "1"]),
            "byOpponent": {
                opponent["name"]: metrics(
                    [row for row in frozen_rows if row.get("opponent") == opponent["name"]]
                )
                for opponent in frozen
            },
        }

    report = {
        **metadata,
        "status": "complete" if not failed_shards and len(rows) == len(schedule) else "partial",
        "completedAt": now().isoformat(),
        "resultRows": len(rows),
        "failedShards": failed_shards,
        "aggregate": metrics(rows),
        "chains": chains,
    }
    write(round_root / "report.json", report)
    write(EVAL_ROOT / "latest.json", {**report, "roundPath": str(round_root)})
    print(json.dumps({**report, "roundPath": str(round_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
