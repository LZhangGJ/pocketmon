from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from common import canonical_deck_sha256, read_deck

ALIASES = {
    "alakazam": "A03",
    "grimmsnarl_froslass_munkidori": "A02",
    "dragapult": "A06",
    "mega_lucario": "LUCARIO",
    "mega_lucario_ex": "LUCARIO",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def state_lock(path: Path) -> Iterator[None]:
    """Serialize league publication. Production callers run on shared Linux storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except ImportError:
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def canonical_archetype(row: dict[str, Any]) -> str:
    explicit = row.get("canonical_archetype")
    if explicit:
        return str(explicit).upper()
    raw = str(row.get("archetype", "unknown"))
    return ALIASES.get(raw.lower(), raw.upper())


def build_pool_payload(league: dict[str, Any]) -> dict[str, Any]:
    base_path = Path(league["basePool"]["path"])
    base = read_json(base_path)
    base_agents = base.get("agents")
    if not isinstance(base_agents, list) or not base_agents:
        raise ValueError("base pool has no agents")
    enabled_dynamic_chains = {
        chain_name
        for chain_name, chain in league["chains"].items()
        if bool(chain.get("poolControl", {}).get("enabled", True))
        and bool(chain.get("current", {}).get("packageManifest"))
    }

    # A base row may be a rollback fallback for a live PPO chain.  Keep it in
    # the immutable base manifest, but suppress it while that chain has a
    # deployable current snapshot.  This makes generation publication a true
    # replacement rather than an ever-growing append-only pool.
    agents = []
    identity_to_index: dict[tuple[str, str], int] = {}

    def add_or_replace(row: dict[str, Any]) -> None:
        normalized = {**row, "canonical_archetype": canonical_archetype(row)}
        deck_sha = str(
            normalized.get("deck_canonical_sha256")
            or normalized.get("deckSha256")
            or ""
        )
        directory_sha = str(
            normalized.get("directory_sha256")
            or normalized.get("directorySha256")
            or ""
        )
        identity = (deck_sha, directory_sha)
        if all(identity):
            existing = identity_to_index.get(identity)
            if existing is not None:
                agents[existing] = normalized
                return
            identity_to_index[identity] = len(agents)
        agents.append(normalized)

    for row in base_agents:
        replacement_chain = str(row.get("replacement_chain", ""))
        if replacement_chain and replacement_chain in enabled_dynamic_chains:
            continue
        add_or_replace(row)
    dynamic = []
    for chain_name, chain in sorted(league["chains"].items()):
        if not bool(chain.get("poolControl", {}).get("enabled", True)):
            continue
        current = chain.get("current", {})
        manifest_path = current.get("packageManifest")
        if not manifest_path:
            continue
        manifest = read_json(Path(manifest_path))
        packages = manifest.get("packages", [])
        if chain.get("learnerDeckPool"):
            candidates = list(packages)
            if not candidates:
                raise ValueError(f"universal chain has no deployable packages: {manifest_path}")
        else:
            candidates = [
                row
                for row in packages
                if str(row.get("archetypeId", "")).upper()
                == str(chain["archetypeId"]).upper()
            ]
            if len(candidates) != 1:
                raise ValueError(f"expected one deployable package for {chain_name}: {manifest_path}")
        for package in candidates:
            agent_dir = Path(package["agentDir"])
            for required in (agent_dir / "main.py", agent_dir / "deck.csv"):
                if not required.is_file():
                    raise FileNotFoundError(required)
            package_archetype = str(package.get("archetypeId") or chain["archetypeId"])
            package_label = str(package.get("archetypeLabel") or chain["archetypeLabel"])
            row = {
                "name": str(package["name"]),
                "agent_dir": str(agent_dir.resolve()),
                "status": "accepted",
                "pool_status": "live_async_ppo_snapshot",
                "archetype": package_archetype,
                "canonical_archetype": package_archetype.upper(),
                "archetype_label": package_label,
                "deck_canonical_sha256": str(package["deckSha256"]),
                "directory_sha256": str(package["directorySha256"]),
                "skill_tier": "live_ppo",
                "policy_weight_within_archetype": 1.0,
                "ppo_chain": chain_name,
                "ppo_generation": int(current["generation"]),
                "behavior_checkpoint_sha256": str(current["sha256"]),
                "pool_control": "atomic_generation_replace",
            }
            replay_aux = current.get("replayAuxVersion")
            if isinstance(replay_aux, dict):
                row.update(
                    {
                        "replay_aux_version": int(replay_aux["version"]),
                        "replay_aux_base_snapshot_id": str(replay_aux["baseSnapshotId"]),
                        "replay_aux_base_checkpoint_sha256": str(
                            replay_aux["baseCheckpointSha256"]
                        ),
                        "pool_control": "atomic_replay_aux_replace",
                    }
                )
            add_or_replace(row)
            dynamic.append(row["name"])
    result = {key: value for key, value in base.items() if key != "agents"}
    result["agents"] = agents
    result["asyncLeague"] = {
        "createdAt": utc_now(),
        "basePool": {"path": str(base_path.resolve()), "sha256": sha256_file(base_path)},
        "dynamicAgents": dynamic,
        "sampling": "uniform canonical archetype, then uniform agent within archetype",
    }
    return result


def publish_snapshot(
    league_path: Path,
    chain_name: str,
    generation: int,
    checkpoint: Path,
    package_manifest: Path,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    package_manifest = package_manifest.resolve()
    checkpoint_sha = sha256_file(checkpoint)
    package_payload = read_json(package_manifest)
    cohort_receipt = package_payload.get("deckCohortReceipt")
    lock_path = league_path.with_suffix(league_path.suffix + ".lock")
    with state_lock(lock_path):
        league = read_json(league_path)
        if chain_name not in league["chains"]:
            raise KeyError(chain_name)
        chain = league["chains"][chain_name]
        previous = chain.get("current")
        if previous and generation < int(previous["generation"]):
            raise ValueError(
                f"generation cannot move backward for {chain_name}: "
                f"{generation} < {previous['generation']}"
            )
        if previous and generation == int(previous["generation"]):
            if checkpoint_sha != str(previous["sha256"]):
                raise ValueError("same-generation bootstrap checkpoint SHA mismatch")
            if previous.get("packageManifest"):
                raise ValueError("same-generation snapshot is already deployed")
            snapshot = {
                **previous,
                "packageManifest": str(package_manifest),
                "publishedAt": utc_now(),
            }
            if cohort_receipt:
                snapshot["deckCohortReceipt"] = str(Path(cohort_receipt).resolve())
            chain["current"] = snapshot
            league["updatedAt"] = utc_now()
            pool_path = Path(league["poolPath"])
            atomic_write_json(pool_path, build_pool_payload(league))
            league["poolSha256"] = sha256_file(pool_path)
            atomic_write_json(league_path, league)
            return snapshot
        snapshot = {
            "generation": generation,
            "checkpoint": str(checkpoint),
            "sha256": checkpoint_sha,
            "snapshotId": f"{chain_name}-g{generation:06d}-{checkpoint_sha[:12]}",
            "packageManifest": str(package_manifest),
            "publishedAt": utc_now(),
        }
        if cohort_receipt:
            snapshot["deckCohortReceipt"] = str(Path(cohort_receipt).resolve())
        history = list(chain.get("history", []))
        if previous:
            history.append(previous)
        chain["history"] = history[-64:]
        chain["current"] = snapshot
        league["updatedAt"] = utc_now()
        pool_path = Path(league["poolPath"])
        atomic_write_json(pool_path, build_pool_payload(league))
        league["poolSha256"] = sha256_file(pool_path)
        atomic_write_json(league_path, league)
    return snapshot


def initialize(league_path: Path, config_path: Path) -> None:
    if league_path.exists():
        raise FileExistsError(league_path)
    config = read_json(config_path)
    chains = config.get("chains", {})
    if not isinstance(chains, dict) or not chains:
        raise ValueError("asynchronous league requires at least one PPO chain")
    for name, chain in chains.items():
        checkpoint = Path(chain["current"]["checkpoint"])
        chain["current"]["sha256"] = sha256_file(checkpoint)
        chain["current"]["snapshotId"] = (
            f"{name}-g{int(chain['current']['generation']):06d}-"
            f"{chain['current']['sha256'][:12]}"
        )
        chain.setdefault("history", [])
    config["schemaVersion"] = 1
    config["createdAt"] = utc_now()
    config["updatedAt"] = config["createdAt"]
    atomic_write_json(Path(config["poolPath"]), build_pool_payload(config))
    config["poolSha256"] = sha256_file(Path(config["poolPath"]))
    atomic_write_json(league_path, config)


def add_chain(league_path: Path, chain_name: str, chain_path: Path) -> dict[str, Any]:
    """Atomically add a bootstrapped-but-not-yet-deployed PPO chain to a live league."""
    incoming = read_json(chain_path)
    required = {
        "deckName",
        "archetypeId",
        "archetypeLabel",
        "deckPath",
        "deckSha256",
        "teacher",
        "current",
    }
    missing = sorted(required - set(incoming))
    if missing:
        raise ValueError(f"chain config is missing required fields: {missing}")
    current = incoming.get("current")
    if not isinstance(current, dict) or "generation" not in current or "checkpoint" not in current:
        raise ValueError("chain current must contain generation and checkpoint")
    deck_path = Path(incoming["deckPath"]).resolve()
    checkpoint = Path(current["checkpoint"]).resolve()
    teacher = Path(incoming["teacher"]).resolve()
    for required_path in (deck_path, checkpoint, teacher):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    actual_deck_sha = canonical_deck_sha256(read_deck(deck_path))
    if actual_deck_sha != str(incoming["deckSha256"]):
        raise ValueError(
            f"deck SHA mismatch for {chain_name}: {actual_deck_sha} != {incoming['deckSha256']}"
        )
    checkpoint_sha = sha256_file(checkpoint)
    normalized = {
        **incoming,
        "deckPath": str(deck_path),
        "teacher": str(teacher),
        "current": {
            **current,
            "generation": int(current["generation"]),
            "checkpoint": str(checkpoint),
            "sha256": checkpoint_sha,
            "snapshotId": (
                f"{chain_name}-g{int(current['generation']):06d}-{checkpoint_sha[:12]}"
            ),
        },
        "history": list(incoming.get("history", [])),
    }
    lock_path = league_path.with_suffix(league_path.suffix + ".lock")
    with state_lock(lock_path):
        league = read_json(league_path)
        if chain_name in league["chains"]:
            raise ValueError(f"PPO chain already exists: {chain_name}")
        league["chains"][chain_name] = normalized
        league["updatedAt"] = utc_now()
        pool_path = Path(league["poolPath"])
        atomic_write_json(pool_path, build_pool_payload(league))
        league["poolSha256"] = sha256_file(pool_path)
        atomic_write_json(league_path, league)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage an asynchronous multi-policy PPO league")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("initialize")
    init.add_argument("--league", type=Path, required=True)
    init.add_argument("--config", type=Path, required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("--league", type=Path, required=True)
    publish.add_argument("--chain", required=True)
    publish.add_argument("--generation", type=int, required=True)
    publish.add_argument("--checkpoint", type=Path, required=True)
    publish.add_argument("--package-manifest", type=Path, required=True)
    add = sub.add_parser("add-chain")
    add.add_argument("--league", type=Path, required=True)
    add.add_argument("--name", required=True)
    add.add_argument("--chain-config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "initialize":
        initialize(args.league.resolve(), args.config.resolve())
    elif args.command == "publish":
        result = publish_snapshot(
            args.league.resolve(),
            args.chain,
            args.generation,
            args.checkpoint,
            args.package_manifest,
        )
        print(json.dumps(result, ensure_ascii=False))
    else:
        result = add_chain(
            args.league.resolve(),
            args.name,
            args.chain_config.resolve(),
        )
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
