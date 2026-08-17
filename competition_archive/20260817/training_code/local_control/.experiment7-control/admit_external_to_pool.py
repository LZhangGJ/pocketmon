from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def load_smoke_rows(paths: list[Path], learner: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("learner") != learner:
                    raise ValueError(f"unexpected learner in smoke result {path}: {row.get('learner')}")
                key = tuple(row.get(field, "") for field in ("learner", "opponent", "seed", "learner_seat"))
                if key in seen:
                    raise ValueError(f"duplicate smoke result key: {key}")
                seen.add(key)
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Admit one smoke-tested external Agent into a new frozen pool")
    parser.add_argument("--base-pool", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--staging-manifest", type=Path, required=True)
    parser.add_argument("--smoke-summary", type=Path, required=True)
    parser.add_argument("--smoke-results", type=Path, nargs="+", required=True)
    parser.add_argument("--reverify-receipt", type=Path, required=True)
    parser.add_argument("--fixed-commit", required=True)
    parser.add_argument("--skill-tier", default="hard")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    base_path = args.base_pool.resolve()
    staging_path = args.staging_manifest.resolve()
    smoke_path = args.smoke_summary.resolve()
    reverify_path = args.reverify_receipt.resolve()
    result_paths = [path.resolve() for path in args.smoke_results]
    output = args.output.resolve()
    receipt = args.receipt.resolve()
    if output.exists() or receipt.exists():
        raise FileExistsError(f"refusing to overwrite v3 pool artifacts: {output} / {receipt}")
    expected_base_hash = args.expected_base_sha256.lower()
    actual_base_hash = sha256_file(base_path)
    if actual_base_hash != expected_base_hash:
        raise ValueError(f"base pool hash mismatch: expected={expected_base_hash} actual={actual_base_hash}")

    base = json.loads(base_path.read_text(encoding="utf-8"))
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    reverify = json.loads(reverify_path.read_text(encoding="utf-8"))
    staged_agents = staging.get("agents", [])
    challengers = smoke.get("challengers", [])
    if len(staged_agents) != 1 or len(challengers) != 1:
        raise ValueError("admission requires exactly one staged Agent and one smoke challenger")
    staged = staged_agents[0]
    challenger = challengers[0]
    name = str(staged["name"])
    if challenger.get("learner") != name:
        raise ValueError("staging and smoke Agent names do not match")
    if staging.get("externalAgentCodeExecuted") is not False:
        raise ValueError("static staging receipt does not prove non-execution")
    if challenger.get("games") != 20 or challenger.get("failures") != 0:
        raise ValueError("external Agent did not complete the 20-game zero-failure smoke gate")
    if not challenger.get("passesRuntimeGate"):
        raise ValueError("external Agent did not pass the runtime gate")
    if any(challenger["seats"][str(seat)].get("games") != 10 for seat in (0, 1)):
        raise ValueError("external Agent smoke is not seat-balanced 10/10")
    if reverify.get("externalAgentCodeExecuted") is not False:
        raise ValueError("post-smoke reverify receipt is invalid")
    reverified_agents = int(reverify.get("agents", 0))
    if reverified_agents != 1:
        raise ValueError("post-smoke reverify receipt did not validate exactly one Agent")

    smoke_rows = load_smoke_rows(result_paths, name)
    if len(smoke_rows) != 20:
        raise ValueError(f"expected 20 unique smoke rows, found {len(smoke_rows)}")
    latencies = [float(row["latency_ms"]) for row in smoke_rows]
    if any(row.get("result") not in {"win", "loss", "draw"} for row in smoke_rows):
        raise ValueError("smoke results contain a failure result")

    agent_row = {
        "name": name,
        "agent_dir": staged["agentDir"],
        "status": "accepted",
        "pool_status": "admitted_screened",
        "archetype": staged["archetype"],
        "deck_canonical_sha256": staged["deckCanonicalSha256"],
        "directory_sha256": staged["directorySha256"],
        "skill_tier": args.skill_tier,
        "policy_weight_within_archetype": 0.0,
        "screening": {
            "games": challenger["games"],
            "wins": challenger["wins"],
            "losses": challenger["losses"],
            "draws": challenger["draws"],
            "failures": challenger["failures"],
            "score_rate": challenger["scoreRate"],
            "wilson95": challenger["wilson95"],
            "seat0_score_rate": challenger["seats"]["0"]["scoreRate"],
            "seat1_score_rate": challenger["seats"]["1"]["scoreRate"],
            "game_latency_ms_p50": percentile(latencies, 0.50),
            "game_latency_ms_p95": percentile(latencies, 0.95),
            "game_latency_ms_max": max(latencies),
        },
        "source": {
            "archive_sha256": staged["sourceArchiveSha256"],
            "static_manifest": str(staging_path),
            "smoke_summary": str(smoke_path),
            "post_smoke_reverify_receipt": str(reverify_path),
            "fixed_commit": args.fixed_commit,
            "network_isolation": "bubblewrap_unshare_net_ro_root",
        },
    }
    agents = list(base.get("agents", []))
    if any(row.get("name") == name for row in agents):
        raise ValueError(f"base pool already contains Agent: {name}")
    agents.append(agent_row)
    by_archetype: dict[str, list[dict[str, Any]]] = {}
    for row in agents:
        by_archetype.setdefault(str(row["archetype"]), []).append(row)
    for rows in by_archetype.values():
        weight = 1.0 / len(rows)
        for row in rows:
            row["policy_weight_within_archetype"] = weight

    created_at = datetime.now(timezone.utc).isoformat()
    v3 = dict(base)
    v3.update(
        {
            "schema": "experiment7_frozen_opponent_pool_v3",
            "schemaVersion": 3,
            "createdAt": created_at,
            "fixedCommit": args.fixed_commit,
            "status": "admitted",
            "agents": agents,
        }
    )
    v3["sources"] = {
        "previousPool": {"path": str(base_path), "sha256": actual_base_hash},
        "newCandidate": {
            "staticManifest": {"path": str(staging_path), "sha256": sha256_file(staging_path)},
            "smokeSummary": {"path": str(smoke_path), "sha256": sha256_file(smoke_path)},
            "postSmokeReverifyReceipt": {
                "path": str(reverify_path),
                "sha256": sha256_file(reverify_path),
            },
            "smokeResults": [
                {"path": str(path), "sha256": sha256_file(path)} for path in result_paths
            ],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(v3, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    output_hash = sha256_file(output)
    audit = {
        "schemaVersion": 1,
        "createdAt": created_at,
        "basePool": {"path": str(base_path), "sha256": actual_base_hash, "agents": len(agents) - 1},
        "newPool": {"path": str(output), "sha256": output_hash, "agents": len(agents)},
        "admittedAgent": {
            "name": name,
            "directorySha256": staged["directorySha256"],
            "deckCanonicalSha256": staged["deckCanonicalSha256"],
            "smokeScoreRate": challenger["scoreRate"],
            "smokeFailures": challenger["failures"],
        },
        "basePoolModified": False,
    }
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
