from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import random
import shlex
import subprocess
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FAILURE_RESULTS = {"crash", "timeout", "illegal"}
EVALUATION_DESIGN_VERSION = 2


def now() -> datetime:
    return datetime.now(timezone.utc)


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write(temporary, payload)
    os.replace(temporary, path)


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def metrics(rows: list[dict[str, str]], invert: bool = False) -> dict[str, Any]:
    result_values = [row.get("result") for row in rows]
    wins = sum(value == ("loss" if invert else "win") for value in result_values)
    losses = sum(value == ("win" if invert else "loss") for value in result_values)
    draws = sum(value == "draw" for value in result_values)
    failures = sum(value in FAILURE_RESULTS for value in result_values)
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


def package_agent(package: dict[str, Any], *, tier: str, status: str) -> dict[str, Any]:
    return {
        "name": package["name"],
        "agent_dir": package["agentDir"],
        "status": "accepted",
        "pool_status": status,
        "archetype": package.get("archetypeId", "unknown"),
        "canonical_archetype": str(package.get("archetypeId", "unknown")).upper(),
        "archetype_label": package.get("archetypeLabel", "unknown"),
        "deck_canonical_sha256": package.get("deckSha256", ""),
        "directory_sha256": package.get("directorySha256", ""),
        "skill_tier": tier,
    }


def ensure_bc_packages(
    *, eval_root: Path, league: dict[str, Any], worktree: Path, python: Path, portable: Path
) -> dict[str, Any]:
    baseline = eval_root / "universal-bc-baseline"
    manifest = baseline / "packages" / "packages.json"
    if manifest.is_file():
        return load(manifest)
    baseline.mkdir(parents=True, exist_ok=True)
    decks = {
        "schemaVersion": 1,
        "selected": [
            {
                "name": chain["deckName"],
                "archetypeId": chain["archetypeId"],
                "archetypeLabel": chain["archetypeLabel"],
                "deckPath": chain["deckPath"],
                "deckSha256": chain["deckSha256"],
            }
            for chain in league["chains"].values()
        ],
    }
    decks_path = baseline / "decks.json"
    write(decks_path, decks)
    subprocess.run(
        [
            str(python),
            str(worktree / "experiment7/integration/export_and_package.py"),
            "package-universal",
            "--reference-root",
            str(Path(league["referenceRoot"])),
            "--sources",
            str(Path(league["sources"])),
            "--decks",
            str(decks_path),
            "--portable",
            str(portable),
            "--output-root",
            str(baseline / "packages"),
            "--name-prefix",
            "universal_bc_baseline",
        ],
        cwd=worktree,
        check=True,
    )
    return load(manifest)


def paired_games(
    schedule: list[dict[str, Any]], learner: str, opponent: str, games: int, seed_base: int
) -> None:
    if games < 2 or games % 2:
        raise ValueError("paired games must be a positive even number")
    for pair_index in range(games // 2):
        seed = seed_base + pair_index
        schedule.append({"learner": learner, "opponent": opponent, "seed": seed, "learner_seat": 0})
        schedule.append({"learner": learner, "opponent": opponent, "seed": seed, "learner_seat": 1})


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the latest four-PPO, frozen-agent, and BC matrix")
    parser.add_argument("--league-root", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--run-shard", type=Path, required=True)
    parser.add_argument("--bc-portable", type=Path, required=True)
    parser.add_argument("--games-per-frozen", type=int, default=4)
    parser.add_argument("--games-per-head-to-head", type=int, default=20)
    parser.add_argument("--shards", type=int, default=48)
    parser.add_argument(
        "--distributed-hosts",
        help="comma-separated directly reachable SSH hosts; omit to run shards locally",
    )
    parser.add_argument("--max-shards-per-host", type=int, default=3)
    args = parser.parse_args()

    league_root = args.league_root.resolve()
    eval_root = league_root / "monitoring" / "full-matrix"
    rounds_root = eval_root / "rounds"
    rounds_root.mkdir(parents=True, exist_ok=True)
    lock_handle = (eval_root / "eval.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        latest = eval_root / "latest.json"
        payload = load(latest) if latest.exists() else {"status": "busy"}
        print(json.dumps({**payload, "busy": True}, ensure_ascii=False))
        return 0

    started = now()
    league = load(league_root / "state" / "league.json")
    base_pool = load(Path(league["basePool"]["path"]))
    frozen = [row for row in base_pool.get("agents", []) if row.get("status", "accepted") == "accepted"]
    if not frozen:
        raise ValueError("base pool has no frozen agents")

    ppo_packages: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for chain_name, chain in sorted(league["chains"].items()):
        current = chain["current"]
        manifest_path = current.get("packageManifest")
        if not manifest_path:
            raise RuntimeError(f"chain has no deployed package: {chain_name}")
        package_rows = load(Path(manifest_path)).get("packages", [])
        if len(package_rows) != 1:
            raise ValueError(f"expected one PPO package for {chain_name}")
        package = package_rows[0]
        ppo_packages.append(package)
        selected.append(
            {
                "chain": chain_name,
                "generation": int(current["generation"]),
                "snapshotId": current["snapshotId"],
                "learner": package["name"],
                "deckSha256": chain["deckSha256"],
                "archetypeId": chain["archetypeId"],
            }
        )
    bc_manifest = ensure_bc_packages(
        eval_root=eval_root,
        league=league,
        worktree=args.worktree.resolve(),
        python=args.python.resolve(),
        portable=args.bc_portable.resolve(),
    )
    bc_by_deck = {row["deckSha256"]: row for row in bc_manifest.get("packages", [])}
    for row in selected:
        if row["deckSha256"] not in bc_by_deck:
            raise ValueError(f"Universal BC package missing for {row['chain']}")
        row["bcLearner"] = bc_by_deck[row["deckSha256"]]["name"]

    generation_tag = "-".join(f"{row['chain']}-g{row['generation']:06d}" for row in selected)
    round_id = f"{started.strftime('%Y%m%dT%H%M%SZ')}-{generation_tag}"
    staging = rounds_root / f".{round_id}.in-progress"
    final = rounds_root / round_id
    staging.mkdir()
    metadata = {
        "schemaVersion": 1,
        "evaluationDesignVersion": EVALUATION_DESIGN_VERSION,
        "status": "running",
        "roundId": round_id,
        "startedAt": started.isoformat(),
        "selected": selected,
        "frozenAgents": [row["name"] for row in frozen],
        "gamesPerFrozen": args.games_per_frozen,
        "gamesPerHeadToHead": args.games_per_head_to_head,
    }
    write(staging / "metadata.json", metadata)

    learner_packages = [*ppo_packages, *[bc_by_deck[row["deckSha256"]] for row in selected]]
    learner_rows = [
        {
            "name": row["name"],
            "agent_dir": row["agentDir"],
            "status": "accepted",
            "directorySha256": row.get("directorySha256", ""),
            "deckSha256": row.get("deckSha256", ""),
            "archetypeId": row.get("archetypeId", ""),
            "archetypeLabel": row.get("archetypeLabel", ""),
        }
        for row in learner_packages
    ]
    write(staging / "learners.json", {"schemaVersion": 1, "agents": learner_rows})
    opponent_rows = [
        *frozen,
        *[package_agent(row, tier="live_ppo", status="live_eval_snapshot") for row in ppo_packages],
        *[package_agent(bc_by_deck[row["deckSha256"]], tier="bc", status="fixed_bc_baseline") for row in selected],
    ]
    write(staging / "opponents.json", {"schemaVersion": 1, "agents": opponent_rows})

    schedule: list[dict[str, Any]] = []
    # Keep evaluation deals fixed across rounds.  PPO and its same-deck BC
    # baseline also receive the same paired-seat seeds against each frozen
    # opponent, so checkpoint deltas are not confounded by a new set of deals.
    seed_cursor = 260_811_000
    for chain_index, selected_row in enumerate(selected):
        ppo_name = selected_row["learner"]
        bc_name = selected_row["bcLearner"]
        for opponent_index, opponent in enumerate(frozen):
            pair_seed = seed_cursor + chain_index * 10_000_000 + opponent_index * 10_000
            paired_games(schedule, ppo_name, opponent["name"], args.games_per_frozen, pair_seed)
            paired_games(schedule, bc_name, opponent["name"], args.games_per_frozen, pair_seed)
        paired_games(
            schedule,
            ppo_name,
            bc_name,
            args.games_per_head_to_head,
            seed_cursor + 500_000_000 + chain_index * 10_000,
        )
    for first_index, first in enumerate(selected):
        for second_index in range(first_index + 1, len(selected)):
            second = selected[second_index]
            paired_games(
                schedule,
                first["learner"],
                second["learner"],
                args.games_per_head_to_head,
                seed_cursor + 700_000_000 + first_index * 1_000_000 + second_index * 10_000,
            )
    random.Random(seed_cursor).shuffle(schedule)
    schedule_path = staging / "schedule.csv"
    with schedule_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("learner", "opponent", "seed", "learner_seat"))
        writer.writeheader()
        writer.writerows(schedule)

    raw = staging / "raw"
    logs = staging / "logs"
    raw.mkdir()
    logs.mkdir()
    requested_hosts = [
        host.strip() for host in (args.distributed_hosts or "").split(",") if host.strip()
    ]
    available_hosts: list[str] = []
    if requested_hosts:
        def probe(host: str) -> tuple[str, bool]:
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

        with ThreadPoolExecutor(max_workers=len(requested_hosts)) as executor:
            available_hosts = [host for host, ok in executor.map(probe, requested_hosts) if ok]
        if not available_hosts:
            raise RuntimeError("no distributed Arena hosts are reachable")
        shard_capacity = len(available_hosts) * args.max_shards_per_host
    else:
        shard_capacity = args.shards
    shard_count = min(args.shards, len(schedule), shard_capacity)
    metadata["distributedHosts"] = available_hosts
    metadata["maxShardsPerHost"] = args.max_shards_per_host if available_hosts else None
    write(staging / "metadata.json", metadata)
    processes: list[tuple[str, subprocess.Popen[str], Any]] = []

    def shard_arguments(shard_index: int) -> list[str]:
        return [
            str(args.worktree.resolve()),
            str(args.python.resolve()),
            str(schedule_path),
            str(staging / "learners.json"),
            str(staging / "opponents.json"),
            str(Path(league["cgDir"])),
            str(raw / f"results-shard-{shard_index:03d}.csv"),
            str(shard_index),
            str(shard_count),
            str(eval_root),
        ]

    if available_hosts:
        for host_index, host in enumerate(available_hosts):
            indices = list(range(host_index, shard_count, len(available_hosts)))
            if not indices:
                continue
            log_handle = (logs / f"host-{host.replace('.', '-')}.log").open("w", encoding="utf-8")
            commands = []
            for shard_index in indices:
                command = ["/homes/lzhang/run_load_guarded_arena_shard.sh", *shard_arguments(shard_index)]
                commands.append(f"{shlex.join(command)} & pids=\"$pids $!\"")
            remote_script = (
                "set -u; pids=\"\"; " + "; ".join(commands)
                + "; failed=0; for pid in $pids; do wait $pid || failed=1; done; exit $failed"
            )
            command = [
                "ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=12",
                f"lzhang@{host}", remote_script,
            ]
            processes.append(
                (host, subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True), log_handle)
            )
    else:
        for shard_index in range(shard_count):
            log_handle = (logs / f"shard-{shard_index:03d}.log").open("w", encoding="utf-8")
            command = ["bash", str(args.run_shard.resolve()), *shard_arguments(shard_index)]
            processes.append(
                (
                    str(shard_index),
                    subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True),
                    log_handle,
                )
            )
    failed_shards = []
    for shard_label, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        if return_code:
            failed_shards.append({"shardOrHost": shard_label, "returnCode": return_code})
    if failed_shards:
        raise RuntimeError(f"Arena shard failures: {failed_shards}")

    rows = read_rows(sorted(raw.glob("results-shard-*.csv")))
    if len(rows) != len(schedule):
        raise RuntimeError(f"Arena coverage mismatch: {len(rows)} != {len(schedule)}")
    frozen_names = {row["name"] for row in frozen}
    previous_path = eval_root / "latest.json"
    previous = load(previous_path) if previous_path.exists() else None
    previous_compatible = bool(
        previous and previous.get("evaluationDesignVersion") == EVALUATION_DESIGN_VERSION
    )
    report: dict[str, Any] = {}
    learner_to_chain = {row["learner"]: row["chain"] for row in selected}
    for selected_row in selected:
        chain = selected_row["chain"]
        ppo_name = selected_row["learner"]
        bc_name = selected_row["bcLearner"]
        agent_rows = []
        for opponent in frozen:
            ppo_metric = metrics([row for row in rows if row["learner"] == ppo_name and row["opponent"] == opponent["name"]])
            bc_metric = metrics([row for row in rows if row["learner"] == bc_name and row["opponent"] == opponent["name"]])
            previous_agent = None
            if previous_compatible:
                previous_agent = next(
                    (item for item in previous.get("chains", {}).get(chain, {}).get("agents", []) if item["agent"] == opponent["name"]),
                    None,
                )
            agent_rows.append(
                {
                    "agent": opponent["name"],
                    "archetype": opponent.get("canonical_archetype", opponent.get("archetype", "unknown")),
                    "ppo": ppo_metric,
                    "universalBc": bc_metric,
                    "ppoMinusBc": (
                        ppo_metric["scoreRate"] - bc_metric["scoreRate"]
                        if ppo_metric["scoreRate"] is not None and bc_metric["scoreRate"] is not None
                        else None
                    ),
                    "deltaVsPrevious": (
                        ppo_metric["scoreRate"] - previous_agent["ppo"]["scoreRate"]
                        if previous_agent and ppo_metric["scoreRate"] is not None and previous_agent["ppo"]["scoreRate"] is not None
                        else None
                    ),
                }
            )
        ppo_frozen_rows = [row for row in rows if row["learner"] == ppo_name and row["opponent"] in frozen_names]
        bc_frozen_rows = [row for row in rows if row["learner"] == bc_name and row["opponent"] in frozen_names]
        aggregate = metrics(ppo_frozen_rows)
        bc_aggregate = metrics(bc_frozen_rows)
        previous_aggregate = (
            previous.get("chains", {}).get(chain, {}).get("frozenAggregate")
            if previous_compatible
            else None
        )
        delta_previous = (
            aggregate["scoreRate"] - previous_aggregate["scoreRate"]
            if previous_aggregate and aggregate["scoreRate"] is not None and previous_aggregate["scoreRate"] is not None
            else None
        )
        if delta_previous is None:
            progress = "baseline"
        elif delta_previous > 0.01:
            progress = "improved"
        elif delta_previous < -0.01:
            progress = "regressed"
        else:
            progress = "flat"
        head_to_head: dict[str, Any] = {}
        for other in selected:
            if other["chain"] == chain:
                continue
            direct = [row for row in rows if row["learner"] == ppo_name and row["opponent"] == other["learner"]]
            inverse = [row for row in rows if row["learner"] == other["learner"] and row["opponent"] == ppo_name]
            head_to_head[other["chain"]] = metrics(direct) if direct else metrics(inverse, invert=True)
        report[chain] = {
            "generation": selected_row["generation"],
            "snapshotId": selected_row["snapshotId"],
            "frozenAggregate": aggregate,
            "universalBcFrozenAggregate": bc_aggregate,
            "ppoMinusBc": aggregate["scoreRate"] - bc_aggregate["scoreRate"],
            "deltaVsPrevious": delta_previous,
            "progress": progress,
            "seatMetrics": {
                seat: metrics([row for row in ppo_frozen_rows if row["learner_seat"] == seat])
                for seat in ("0", "1")
            },
            "seatGap": abs(
                metrics([row for row in ppo_frozen_rows if row["learner_seat"] == "0"])["scoreRate"]
                - metrics([row for row in ppo_frozen_rows if row["learner_seat"] == "1"])["scoreRate"]
            ),
            "directVsUniversalBc": metrics(
                [row for row in rows if row["learner"] == ppo_name and row["opponent"] == bc_name]
            ),
            "ppoHeadToHead": head_to_head,
            "agents": agent_rows,
        }

    metadata.update({"status": "complete", "completedAt": now().isoformat(), "games": len(rows)})
    write(staging / "metadata.json", metadata)
    payload = {
        "schemaVersion": 1,
        "evaluationDesignVersion": EVALUATION_DESIGN_VERSION,
        "status": "complete",
        "busy": False,
        "updatedAt": now().isoformat(),
        "roundId": round_id,
        "games": len(rows),
        "engineSeedControlled": all(
            row.get("engine_seed_controlled", "").lower() == "true" for row in rows
        ),
        "seedPolicy": "fixed paired-seat Python agent RNG seeds; native engine deal RNG is uncontrolled",
        "frozenAgentCount": len(frozen),
        "chains": report,
        "previousRoundId": previous.get("roundId") if previous_compatible else None,
    }
    write(staging / "report.json", payload)
    os.replace(staging, final)
    payload["roundPath"] = str(final)
    atomic_write(eval_root / "latest.json", payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise
