from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
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
EVALUATION_DESIGN_VERSION = 4

# Leaderboard archetypes explicitly selected by the user on 2026-08-15.  The
# frozen side of the hourly matrix is the union of *uncovered* families only.
# Current PPOs are always evaluated head-to-head, independently of this list.
IMAGE_FAMILY_ARCHETYPES: dict[str, set[str]] = {
    "dragapult_ex": {"DRAGAPULT"},
    "hydrapple_meganium_ogerpon": {"HYDRAPPLE_OGERPON"},
    "alakazam": {"ALAKAZAM"},
    "mega_froslass_lopunny": {"MEGA_LOPUNNY_MEGA_FROSLASS"},
    "slowking_control": {"KANGASKHAN_SLOWKING"},
    "c20_grimmsnarl_froslass": {"GRIMMSNARL_FROSLASS"},
    "crustle_kangaskhan_control": {"KANGASKHAN_CRUSTLE"},
    "teal_mask_ogerpon_ex": {"OGERPON_TOOLBOX"},
    "arboliva_meganium_ogerpon": {"OGERPON_TOOLBOX"},
    "chandelure_comfey": {"CHANDELURE"},
    "mega_lucario": {"MEGA_LUCARIO"},
    "raging_bolt_ex": {"RAGING_BOLT_OGERPON"},
}

# Some PPO chain IDs are competition-specific aliases rather than the pool's
# canonical archetype IDs.  These aliases make family coverage explicit.
PPO_FAMILY_ALIASES: dict[str, set[str]] = {
    "dragapult_ex": {"A06", "DRAGAPULT"},
    "c20_grimmsnarl_froslass": {"A02", "GRIMMSNARL_FROSLASS"},
    "teal_mask_ogerpon_ex": {"TEAL_MASK_OGERPON", "OGERPON_TOOLBOX"},
    "arboliva_meganium_ogerpon": {
        "ARBOLIVA_MEGANIUM_OGERPON",
        "OGERPON_TOOLBOX",
    },
    "mega_lucario": {"LUCARIO_GOLD", "MEGA_LUCARIO"},
}


def image_family_coverage(league: dict[str, Any]) -> tuple[list[str], list[str]]:
    ppo_archetypes = {
        str(chain.get("archetypeId", "")).upper()
        for chain in league.get("chains", {}).values()
    }
    covered: list[str] = []
    uncovered: list[str] = []
    for family, pool_archetypes in IMAGE_FAMILY_ARCHETYPES.items():
        aliases = PPO_FAMILY_ALIASES.get(family, pool_archetypes)
        (covered if ppo_archetypes.intersection(aliases) else uncovered).append(family)
    return covered, uncovered


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


def select_or_build_ppo_package(
    *,
    chain_name: str,
    chain: dict[str, Any],
    package_rows: list[dict[str, Any]],
    eval_root: Path,
    league: dict[str, Any],
    worktree: Path,
    python: Path,
) -> dict[str, Any]:
    matching = [row for row in package_rows if row.get("deckSha256") == chain["deckSha256"]]
    if len(matching) == 1:
        return matching[0]
    if len(package_rows) == 1:
        return package_rows[0]
    current = chain["current"]
    manifest_path = Path(current["packageManifest"])
    portable = manifest_path.parent.parent / "universal_ppo.npz"
    if not portable.is_file():
        raise ValueError(
            f"cannot resolve representative package for {chain_name}: "
            f"{len(package_rows)} rows and no portable model at {portable}"
        )
    output_root = eval_root / "derived-ppo-packages" / current["snapshotId"]
    manifest = output_root / "packages.json"
    if not manifest.is_file():
        output_root.mkdir(parents=True, exist_ok=True)
        decks_path = output_root / "representative-deck.json"
        write(
            decks_path,
            {
                "schemaVersion": 1,
                "selected": [
                    {
                        "name": chain["deckName"],
                        "archetypeId": chain["archetypeId"],
                        "archetypeLabel": chain["archetypeLabel"],
                        "deckPath": chain["deckPath"],
                        "deckSha256": chain["deckSha256"],
                    }
                ],
            },
        )
        subprocess.run(
            [
                str(python),
                str(worktree / "experiment7/integration/export_and_package.py"),
                "package-universal",
                "--reference-root", str(Path(league["referenceRoot"])),
                "--sources", str(Path(league["sources"])),
                "--decks", str(decks_path),
                "--portable", str(portable),
                "--output-root", str(output_root),
                "--name-prefix", f"live_{chain_name}_g{int(current['generation']):06d}_representative",
            ],
            cwd=worktree,
            check=True,
        )
    derived = load(manifest).get("packages", [])
    matching = [row for row in derived if row.get("deckSha256") == chain["deckSha256"]]
    if len(matching) != 1:
        raise ValueError(f"failed to build one representative package for {chain_name}")
    return matching[0]


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
    parser = argparse.ArgumentParser(
        description="Run all current PPO head-to-head plus uncovered image-whitelist archetypes"
    )
    parser.add_argument("--league-root", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--run-shard", type=Path, required=True)
    parser.add_argument("--bc-portable", type=Path, required=True)
    parser.add_argument("--games-per-frozen", type=int, default=4)
    parser.add_argument("--games-per-head-to-head", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--timeout-retries", type=int, default=2)
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
    base_pool_path = Path(league["basePool"]["path"])
    base_pool_bytes = base_pool_path.read_bytes()
    base_pool_sha256 = hashlib.sha256(base_pool_bytes).hexdigest()
    base_pool = json.loads(base_pool_bytes)
    source_frozen = [
        row for row in base_pool.get("agents", [])
        if row.get("status", "accepted") == "accepted"
    ]
    if not source_frozen:
        raise ValueError("base pool has no frozen agents")

    ppo_packages: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for chain_name, chain in sorted(league["chains"].items()):
        current = chain["current"]
        manifest_path = current.get("packageManifest")
        if not manifest_path:
            raise RuntimeError(f"chain has no deployed package: {chain_name}")
        package_rows = load(Path(manifest_path)).get("packages", [])
        package = select_or_build_ppo_package(
            chain_name=chain_name,
            chain=chain,
            package_rows=package_rows,
            eval_root=eval_root,
            league=league,
            worktree=args.worktree.resolve(),
            python=args.python.resolve(),
        )
        ppo_packages.append(package)
        selected.append(
            {
                "chain": chain_name,
                "generation": int(current["generation"]),
                "snapshotId": current["snapshotId"],
                "learner": package["name"],
                "deckSha256": chain["deckSha256"],
                "archetypeId": chain["archetypeId"],
                "checkpointSha256": package.get("directorySha256", ""),
            }
        )
    covered_families, uncovered_families = image_family_coverage(league)
    uncovered_archetypes = {
        archetype
        for family in uncovered_families
        for archetype in IMAGE_FAMILY_ARCHETYPES[family]
    }
    frozen = [
        row for row in source_frozen
        if str(row.get("canonical_archetype", row.get("archetype", ""))).upper()
        in uncovered_archetypes
    ]

    generation_tag = "-".join(f"{row['chain']}-g{row['generation']:06d}" for row in selected)
    generation_fingerprint = hashlib.sha256(generation_tag.encode("utf-8")).hexdigest()[:16]
    evaluation_config = {
        "evaluationDesignVersion": EVALUATION_DESIGN_VERSION,
        "gamesPerFrozen": args.games_per_frozen,
        "gamesPerHeadToHead": args.games_per_head_to_head,
        "seedBase": 260_811_000,
        "timeoutSeconds": args.timeout_seconds,
        "timeoutRetries": args.timeout_retries,
        "basePoolSha256": base_pool_sha256,
        "sourceFrozenAgentCount": len(source_frozen),
        "imageWhitelistCoveredFamilies": covered_families,
        "imageWhitelistUncoveredFamilies": uncovered_families,
        "imageWhitelistUncoveredArchetypes": sorted(uncovered_archetypes),
        "frozenAgents": [
            {
                "name": row["name"],
                "directorySha256": row.get("directory_sha256", row.get("directorySha256", "")),
            }
            for row in frozen
        ],
    }
    evaluation_config_sha256 = hashlib.sha256(
        json.dumps(evaluation_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    round_id = (
        f"{started.strftime('%Y%m%dT%H%M%SZ')}-allppo-{len(selected):02d}-"
        f"{generation_fingerprint}"
    )
    staging = rounds_root / f".{round_id}.in-progress"
    final = rounds_root / round_id
    staging.mkdir()
    metadata = {
        "schemaVersion": 1,
        "evaluationDesignVersion": EVALUATION_DESIGN_VERSION,
        "status": "running",
        "roundId": round_id,
        "generationTag": generation_tag,
        "generationFingerprint": generation_fingerprint,
        "basePoolPath": str(base_pool_path),
        "basePoolSha256": base_pool_sha256,
        "sourceFrozenAgentCount": len(source_frozen),
        "imageWhitelistCoveredFamilies": covered_families,
        "imageWhitelistUncoveredFamilies": uncovered_families,
        "imageWhitelistUncoveredArchetypes": sorted(uncovered_archetypes),
        "evaluationConfigSha256": evaluation_config_sha256,
        "startedAt": started.isoformat(),
        "selected": selected,
        "frozenAgents": [row["name"] for row in frozen],
        "gamesPerFrozen": args.games_per_frozen,
        "gamesPerHeadToHead": args.games_per_head_to_head,
        "timeoutSeconds": args.timeout_seconds,
        "timeoutRetries": args.timeout_retries,
    }
    write(staging / "metadata.json", metadata)

    learner_packages = ppo_packages
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
    ]
    write(staging / "opponents.json", {"schemaVersion": 1, "agents": opponent_rows})

    schedule: list[dict[str, Any]] = []
    # Keep evaluation deals fixed across rounds so checkpoint deltas are not
    # confounded by a new set of Python-side agent RNG seeds.
    seed_cursor = 260_811_000
    for chain_index, selected_row in enumerate(selected):
        ppo_name = selected_row["learner"]
        for opponent_index, opponent in enumerate(frozen):
            pair_seed = seed_cursor + chain_index * 10_000_000 + opponent_index * 10_000
            paired_games(schedule, ppo_name, opponent["name"], args.games_per_frozen, pair_seed)
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
                command = [
                    "env",
                    "ARENA_CPU_LIMIT_PERCENT=80",
                    "ARENA_IO_LIMIT_PERCENT=20",
                    f"ARENA_TIMEOUT_SECONDS={args.timeout_seconds}",
                    f"ARENA_TIMEOUT_RETRIES={args.timeout_retries}",
                    (
                        "/suedata1/Free/lzhang/pocketmon-runs/"
                        "experiment7-hourly-image-whitelist-v4-20260815/control/"
                        "run_load_guarded_arena_shard_v4_retry.sh"
                    ),
                    *shard_arguments(shard_index),
                ]
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
            command = [
                "env",
                f"ARENA_TIMEOUT_SECONDS={args.timeout_seconds}",
                f"ARENA_TIMEOUT_RETRIES={args.timeout_retries}",
                "bash",
                str(args.run_shard.resolve()),
                *shard_arguments(shard_index),
            ]
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
        previous
        and previous.get("evaluationDesignVersion") == EVALUATION_DESIGN_VERSION
        and previous.get("basePoolSha256") == base_pool_sha256
        and previous.get("evaluationConfigSha256") == evaluation_config_sha256
    )
    report: dict[str, Any] = {}
    learner_to_chain = {row["learner"]: row["chain"] for row in selected}
    for selected_row in selected:
        chain = selected_row["chain"]
        ppo_name = selected_row["learner"]
        agent_rows = []
        for opponent in frozen:
            ppo_metric = metrics([row for row in rows if row["learner"] == ppo_name and row["opponent"] == opponent["name"]])
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
                    "deltaVsPrevious": (
                        ppo_metric["scoreRate"] - previous_agent["ppo"]["scoreRate"]
                        if previous_agent and ppo_metric["scoreRate"] is not None and previous_agent["ppo"]["scoreRate"] is not None
                        else None
                    ),
                }
            )
        ppo_frozen_rows = [row for row in rows if row["learner"] == ppo_name and row["opponent"] in frozen_names]
        aggregate = metrics(ppo_frozen_rows)
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
            "deltaVsPrevious": delta_previous,
            "progress": progress,
            "seatMetrics": {
                seat: metrics([row for row in ppo_frozen_rows if row["learner_seat"] == seat])
                for seat in ("0", "1")
            },
            "seatGap": (
                abs(
                    metrics([row for row in ppo_frozen_rows if row["learner_seat"] == "0"])["scoreRate"]
                    - metrics([row for row in ppo_frozen_rows if row["learner_seat"] == "1"])["scoreRate"]
                )
                if ppo_frozen_rows else None
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
        "basePoolPath": str(base_pool_path),
        "basePoolSha256": base_pool_sha256,
        "sourceFrozenAgentCount": len(source_frozen),
        "imageWhitelistCoveredFamilies": covered_families,
        "imageWhitelistUncoveredFamilies": uncovered_families,
        "imageWhitelistUncoveredArchetypes": sorted(uncovered_archetypes),
        "evaluationConfigSha256": evaluation_config_sha256,
        "games": len(rows),
        "timeoutSeconds": args.timeout_seconds,
        "timeoutRetries": args.timeout_retries,
        "retryAttempts": sum(int(row.get("retry_count") or 0) for row in rows),
        "gamesRecoveredByRetry": sum(
            int(row.get("retry_count") or 0) > 0 and row.get("result") not in FAILURE_RESULTS
            for row in rows
        ),
        "timeoutsAfterRetries": sum(row.get("result") == "timeout" for row in rows),
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
