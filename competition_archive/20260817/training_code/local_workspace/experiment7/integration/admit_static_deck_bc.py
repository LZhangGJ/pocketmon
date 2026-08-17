from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--archetype", required=True)
    parser.add_argument("--archetype-label", required=True)
    parser.add_argument("--deck-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--portable", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--static-pool", type=Path, required=True)
    parser.add_argument("--live-pool", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    parity = load(args.parity.resolve())
    smoke = load(args.smoke.resolve())
    if any(int(parity.get(key, -1)) != 0 for key in (
        "actionMismatches", "stableRankingMismatches", "fullRankingMismatches", "illegalPredictionCount"
    )):
        raise ValueError("portable parity is not exact/ legal")
    if smoke.get("status") != "complete" or not smoke.get("packageImportSmoke") or not smoke.get("diagnosticsCallable"):
        raise ValueError("Agent smoke is not complete")
    checkpoint_sha = sha256_file(args.checkpoint.resolve())
    portable_sha = sha256_file(args.portable.resolve())
    if parity.get("checkpointSha256") != checkpoint_sha or parity.get("portableSha256") != portable_sha:
        raise ValueError("parity receipt SHA does not bind admission artifacts")
    if smoke.get("deckSha256") != args.deck_sha256:
        raise ValueError("Agent smoke deck identity does not match admission")

    policy_version = checkpoint_sha
    name = f"static_bc_{args.profile}_{policy_version[:12]}__{args.deck_sha256[:12]}"
    agent_dir = args.agent_dir.resolve()
    entry = {
        "name": name,
        "agent_dir": str(agent_dir),
        "status": "accepted",
        "pool_status": "static_frozen_replay_bc",
        "archetype": args.archetype,
        "canonical_archetype": args.archetype,
        "archetype_label": args.archetype_label,
        "deck_canonical_sha256": args.deck_sha256,
        "directory_sha256": directory_sha256(agent_dir),
        "skill_tier": "static_frozen",
        "policy_weight_within_archetype": 1.0,
        "sourceKind": "replay_static_bc",
        "staticProfile": args.profile,
        "policyVersion": policy_version,
        "behavior_checkpoint_sha256": checkpoint_sha,
        "portable_sha256": portable_sha,
        "ppoUpdatesAllowed": False,
        "immutable": True,
        "replacementSource": "future_replay_pipeline_only",
        "pool_control": "deck_identity_plus_policy_version_dedup",
    }

    static_path = args.static_pool.resolve()
    static = load(static_path) if static_path.is_file() else {
        "schemaVersion": 1,
        "kind": "experiment7_static_frozen_replay_bc_pool",
        "createdAt": now(),
        "agents": [],
    }
    live_path = args.live_pool.resolve()
    live = load(live_path)
    static_before = len(static["agents"])
    live_before = len(live["agents"])
    dedupe = lambda row: (
        row.get("deck_canonical_sha256") == args.deck_sha256
        and row.get("policyVersion", row.get("behavior_checkpoint_sha256")) == policy_version
    )
    static["agents"] = [row for row in static["agents"] if not dedupe(row)] + [entry]
    live["agents"] = [row for row in live["agents"] if not dedupe(row)] + [entry]
    dynamic = live.setdefault("asyncLeague", {}).setdefault("dynamicAgents", [])
    live["asyncLeague"]["dynamicAgents"] = [value for value in dynamic if value != name] + [name]
    observed = now()
    static["updatedAt"] = observed
    live["updatedAt"] = observed
    atomic_json(static_path, static)
    atomic_json(live_path, live)
    receipt = {
        "schemaVersion": 1,
        "status": "admitted",
        "admittedAt": observed,
        "profile": args.profile,
        "dedupeKey": {"deckSha256": args.deck_sha256, "policyVersion": policy_version},
        "entry": entry,
        "portableParity": {"decisions": parity["decisions"], "mismatches": 0, "illegal": 0},
        "agentSmoke": {"packageImportSmoke": True, "diagnosticsCallable": True},
        "staticPool": {"path": str(static_path), "before": static_before, "after": len(static["agents"]), "sha256": sha256_file(static_path)},
        "livePool": {"path": str(live_path), "before": live_before, "after": len(live["agents"]), "sha256": sha256_file(live_path)},
        "staticFrozen": True,
        "ppoUpdatesAllowed": False,
    }
    atomic_json(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
