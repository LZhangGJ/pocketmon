from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from pool_reconciler import replace_core_and_reconcile


DEFAULT_CHAINS = (
    "a02_grim_large_g9_pokegear",
    "a08_maxbelt_large_g9",
    "dragapult_munkidori_large_g9",
    "lucario_gold_exact",
    "alakazam_large_g9",
    "kangaskhan_crustle_large_g9",
    "festival_grass_large_g9",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@contextmanager
def state_lock(path: Path) -> Iterator[None]:
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


def find_reports(roots: Sequence[Path]) -> list[dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for root in roots:
        paths = [root] if root.is_file() else root.glob("**/report.json")
        for path in paths:
            try:
                report = read_json(path)
            except (FileNotFoundError, json.JSONDecodeError, TypeError):
                continue
            round_id = str(report.get("roundId", ""))
            if report.get("status") != "complete" or not round_id:
                continue
            reports[round_id] = {**report, "_path": str(path.resolve())}
    return sorted(
        reports.values(),
        key=lambda row: (str(row.get("updatedAt", "")), str(row["roundId"])),
    )


def select_package(chain_name: str, chain: dict[str, Any]) -> dict[str, Any]:
    current = chain["current"]
    manifest_path = Path(current["packageManifest"])
    manifest = read_json(manifest_path)
    archetype = str(chain["archetypeId"]).upper()
    packages = [
        row
        for row in manifest.get("packages", [])
        if str(row.get("archetypeId", "")).upper() == archetype
    ]
    if len(packages) != 1:
        raise ValueError(
            f"expected exactly one package for {chain_name}/{archetype}: {manifest_path}"
        )
    package = packages[0]
    agent_dir = Path(package["agentDir"])
    for required in (agent_dir / "main.py", agent_dir / "deck.csv"):
        if not required.is_file():
            raise FileNotFoundError(required)
    gate = dict(
        (chain.get("trainingControl") or {}).get("promotionTacticalGate") or {}
    )
    applies_at = str(gate.get("appliesToPublishedAtOrAfter", ""))
    published_at = str(current.get("publishedAt", ""))
    gate_applies = bool(gate.get("enabled"))
    if gate_applies and applies_at:
        try:
            gate_applies = datetime.fromisoformat(
                published_at.replace("Z", "+00:00")
            ) >= datetime.fromisoformat(applies_at.replace("Z", "+00:00"))
        except ValueError:
            gate_applies = False
    gate["appliesToCandidate"] = gate_applies
    return {
        "chain": chain_name,
        "generation": int(current["generation"]),
        "snapshotId": str(current["snapshotId"]),
        "checkpoint": str(Path(current["checkpoint"]).resolve()),
        "checkpointSha256": str(current["sha256"]),
        "packageManifest": str(manifest_path.resolve()),
        "agentDir": str(agent_dir.resolve()),
        "packageName": str(package["name"]),
        "directorySha256": str(package["directorySha256"]),
        "deckSha256": str(package["deckSha256"]),
        "archetypeId": str(chain["archetypeId"]),
        "archetypeLabel": str(chain["archetypeLabel"]),
        "publishedAt": str(current.get("publishedAt", "")),
        "tacticalGate": gate,
    }


def new_candidate(package: dict[str, Any]) -> dict[str, Any]:
    return {
        **package,
        "registeredAt": utc_now(),
        "status": "testing",
        "passRounds": [],
        "observations": [],
    }


def refresh_candidate_tactical_gate(
    candidate: dict[str, Any], current_gate: dict[str, Any]
) -> None:
    """Attach the current gate policy without retroactively gating old snapshots."""
    gate = dict(candidate.get("tacticalGate") or current_gate or {})
    applies_at = str(gate.get("appliesToPublishedAtOrAfter", ""))
    published_at = str(candidate.get("publishedAt", ""))
    applies = bool(gate.get("enabled"))
    if applies and applies_at:
        try:
            applies = datetime.fromisoformat(
                published_at.replace("Z", "+00:00")
            ) >= datetime.fromisoformat(applies_at.replace("Z", "+00:00"))
        except ValueError:
            applies = False
    gate["appliesToCandidate"] = applies
    candidate["tacticalGate"] = gate


def evaluate_round(
    candidate: dict[str, Any],
    report: dict[str, Any],
    *,
    min_frozen_games: int,
    min_direct_games: int,
    max_regression: float,
    initial_min_score: float,
    tactical_evidence: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    chain_metrics = report.get("chains", {}).get(candidate["chain"])
    if not isinstance(chain_metrics, dict):
        return None
    if str(chain_metrics.get("snapshotId")) != str(candidate["snapshotId"]):
        return None
    frozen = chain_metrics.get("frozenAggregate", {})
    direct = chain_metrics.get("directVsUniversalBc", {})
    delta = chain_metrics.get("deltaVsPrevious")
    reasons: list[str] = []
    if int(frozen.get("completed", 0)) < min_frozen_games:
        reasons.append("insufficient_frozen_games")
    if int(frozen.get("failures", 0)) != 0:
        reasons.append("frozen_failures")
    if int(direct.get("completed", 0)) < min_direct_games:
        reasons.append("insufficient_direct_games")
    if int(direct.get("failures", 0)) != 0:
        reasons.append("direct_failures")
    score = float(frozen.get("scoreRate", 0.0))
    if candidate.get("firstAdmission"):
        if score < initial_min_score:
            reasons.append("initial_score_below_floor")
    elif delta is None:
        if score < initial_min_score:
            reasons.append("initial_score_below_floor")
    elif float(delta) < -max_regression:
        reasons.append("regressed_over_limit")
    tactical_guardrail = None
    gate = candidate.get("tacticalGate") or {}
    if gate.get("appliesToCandidate"):
        tactical_guardrail = (
            ((tactical_evidence or {}).get("snapshots") or {})
            .get(candidate["chain"], {})
            .get(candidate["snapshotId"])
        )
        if not isinstance(tactical_guardrail, dict):
            reasons.append("missing_exact_snapshot_tactical_evidence")
        else:
            if int(tactical_guardrail.get("minimumRevision", 0)) < int(
                gate.get("minimumRevision", 7)
            ):
                reasons.append("tactical_revision_too_old")
            if int(tactical_guardrail.get("episodes", 0)) < int(
                gate.get("minimumEpisodes", 40)
            ):
                reasons.append("insufficient_tactical_episodes")
            if int(tactical_guardrail.get("trackedOpportunities", 0)) < int(
                gate.get("minimumTrackedOpportunities", 10)
            ):
                reasons.append("insufficient_tactical_opportunities")
            error_rate = tactical_guardrail.get("aggregateErrorRate")
            if error_rate is None or float(error_rate) > float(
                gate.get("maximumAggregateErrorRate", 0.35)
            ):
                reasons.append("tactical_error_rate_above_limit")
        key_gate = gate.get("keyMatchupGate") or {}
        if key_gate.get("enabled"):
            key_aggregate = chain_metrics.get("keyFrozenAggregate") or {}
            key_opponent_count = int(chain_metrics.get("keyOpponentCount", 0))
            games_per_opponent = int(key_gate.get("gamesPerOpponent", 40))
            if key_opponent_count <= 0:
                reasons.append("missing_key_matchup_opponents")
            elif int(key_aggregate.get("completed", 0)) < key_opponent_count * games_per_opponent:
                reasons.append("insufficient_key_matchup_games")
            if int(key_aggregate.get("failures", 0)) != 0:
                reasons.append("key_matchup_failures")
            key_delta = chain_metrics.get("keyDeltaVsPrevious")
            max_key_regression = float(
                key_gate.get("maximumAggregateRegressionPp", 2.0)
            ) / 100.0
            if key_delta is not None and float(key_delta) < -max_key_regression:
                reasons.append("key_matchup_aggregate_regressed_over_limit")
            max_agent_regression = float(
                key_gate.get("maximumSingleOpponentRegressionPp", 10.0)
            ) / 100.0
            for agent in chain_metrics.get("agents", []):
                if not agent.get("keyMatchup"):
                    continue
                if int((agent.get("ppo") or {}).get("completed", 0)) < games_per_opponent:
                    reasons.append(f"insufficient_key_games:{agent.get('agent', 'unknown')}")
                if int((agent.get("ppo") or {}).get("failures", 0)) != 0:
                    reasons.append(f"key_failures:{agent.get('agent', 'unknown')}")
                agent_delta = agent.get("deltaVsPrevious")
                if agent_delta is not None and float(agent_delta) < -max_agent_regression:
                    reasons.append(f"key_agent_regressed:{agent.get('agent', 'unknown')}")
    return {
        "roundId": str(report["roundId"]),
        "report": str(report["_path"]),
        "observedAt": utc_now(),
        "passed": not reasons,
        "reasons": reasons,
        "frozenAggregate": frozen,
        "directVsUniversalBc": direct,
        "deltaVsPrevious": delta,
        "seatGap": chain_metrics.get("seatGap"),
        "tacticalGuardrail": tactical_guardrail,
        "keyFrozenAggregate": chain_metrics.get("keyFrozenAggregate"),
        "keyDeltaVsPrevious": chain_metrics.get("keyDeltaVsPrevious"),
    }


def update_candidate(
    candidate: dict[str, Any],
    reports: Sequence[dict[str, Any]],
    *,
    min_frozen_games: int,
    min_direct_games: int,
    max_regression: float,
    initial_min_score: float,
    required_passes: int,
    tactical_evidence: dict[str, Any] | None = None,
) -> None:
    seen = {row["roundId"] for row in candidate.get("observations", [])}
    for report in reports:
        if report["roundId"] in seen:
            continue
        observation = evaluate_round(
            candidate,
            report,
            min_frozen_games=min_frozen_games,
            min_direct_games=min_direct_games,
            max_regression=max_regression,
            initial_min_score=initial_min_score,
            tactical_evidence=tactical_evidence,
        )
        if observation is None:
            continue
        candidate.setdefault("observations", []).append(observation)
        if observation["passed"]:
            candidate.setdefault("passRounds", []).append(observation["roundId"])
        else:
            candidate["passRounds"] = []
        seen.add(observation["roundId"])
    candidate["status"] = (
        "passed" if len(candidate.get("passRounds", [])) >= required_passes else "testing"
    )


def frozen_agent(candidate: dict[str, Any]) -> dict[str, Any]:
    generation = int(candidate["generation"])
    return {
        "name": f"frozen_{candidate['chain']}_g{generation:06d}",
        "agent_dir": candidate["agentDir"],
        "status": "accepted",
        "pool_status": "frozen_ppo_version",
        "archetype": candidate["archetypeId"],
        "canonical_archetype": str(candidate["archetypeId"]).upper(),
        "archetype_label": candidate["archetypeLabel"],
        "deck_canonical_sha256": candidate["deckSha256"],
        "directory_sha256": candidate["directorySha256"],
        "skill_tier": "frozen_ppo",
        "policy_weight_within_archetype": 1.0,
        "ppo_chain": candidate["chain"],
        "ppo_generation": generation,
        "ppo_snapshot_id": candidate["snapshotId"],
        "behavior_checkpoint_sha256": candidate["checkpointSha256"],
        "source_package_manifest": candidate["packageManifest"],
        "screening": {
            "requiredPasses": len(candidate["passRounds"]),
            "roundIds": list(candidate["passRounds"]),
            "latest": candidate.get("observations", [])[-1]
            if candidate.get("observations")
            else None,
        },
    }


def build_frozen_pool(
    source_pool: dict[str, Any], admissions: dict[str, Any]
) -> dict[str, Any]:
    aliases = {
        "alakazam": "A03",
        "grimmsnarl_froslass_munkidori": "A02",
        "dragapult": "A06",
        "mega_lucario": "LUCARIO",
        "mega_lucario_ex": "LUCARIO",
    }
    agents = [
        {
            **row,
            "canonical_archetype": str(
                row.get("canonical_archetype")
                or aliases.get(
                    str(row.get("archetype", "unknown")).lower(),
                    str(row.get("archetype", "unknown")).upper(),
                )
            ).upper(),
        }
        for row in source_pool.get("agents", [])
        if row.get("pool_status") != "frozen_ppo_version"
    ]
    promoted = []
    for chain_name, admission in sorted(admissions.items()):
        if chain_name.startswith("_"):
            continue
        agent = admission["agent"]
        agents.append(agent)
        promoted.append(agent["name"])
    payload = {key: value for key, value in source_pool.items() if key != "agents"}
    payload["agents"] = agents
    payload["ppoFrozenPromotion"] = {
        "updatedAt": utc_now(),
        "sourcePool": admissions.get("_sourcePool"),
        "activeAgents": promoted,
        "policy": "one latest screened frozen PPO version per chain; full history in promotion state",
    }
    return payload


def rebuild_live_pool(league: dict[str, Any], base_pool: dict[str, Any]) -> dict[str, Any]:
    agents = list(base_pool.get("agents", []))
    seen = {
        (str(row.get("deck_canonical_sha256", "")), str(row.get("directory_sha256", "")))
        for row in agents
    }
    dynamic = []
    suppressed = []
    for chain_name, chain in sorted(league["chains"].items()):
        package = select_package(chain_name, chain)
        identity = (package["deckSha256"], package["directorySha256"])
        if identity in seen:
            suppressed.append(package["packageName"])
            continue
        row = {
            "name": package["packageName"],
            "agent_dir": package["agentDir"],
            "status": "accepted",
            "pool_status": "live_async_ppo_snapshot",
            "archetype": package["archetypeId"],
            "canonical_archetype": package["archetypeId"].upper(),
            "archetype_label": package["archetypeLabel"],
            "deck_canonical_sha256": package["deckSha256"],
            "directory_sha256": package["directorySha256"],
            "skill_tier": "live_ppo",
            "policy_weight_within_archetype": 1.0,
            "ppo_chain": chain_name,
            "ppo_generation": package["generation"],
            "behavior_checkpoint_sha256": package["checkpointSha256"],
        }
        agents.append(row)
        seen.add(identity)
        dynamic.append(row["name"])
    payload = {key: value for key, value in base_pool.items() if key != "agents"}
    payload["agents"] = agents
    payload["asyncLeague"] = {
        "createdAt": utc_now(),
        "dynamicAgents": dynamic,
        "suppressedExactFrozenDuplicates": suppressed,
        "sampling": "uniform canonical archetype, then uniform agent within archetype",
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register, gate, and atomically promote packaged PPO snapshots"
    )
    parser.add_argument("--league", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--source-base-pool", type=Path, required=True)
    parser.add_argument("--output-base-pool", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, action="append", default=[])
    parser.add_argument("--chain", action="append", default=[])
    parser.add_argument("--min-frozen-games", type=int, default=40)
    parser.add_argument("--min-direct-games", type=int, default=40)
    parser.add_argument("--max-regression-pp", type=float, default=2.0)
    parser.add_argument("--initial-min-score", type=float, default=0.5)
    parser.add_argument("--required-passes", type=int, default=2)
    parser.add_argument(
        "--tactical-evidence",
        type=Path,
        help="exact-snapshot revision-7 opportunity/error evidence",
    )
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument(
        "--reset-admissions",
        action="store_true",
        help="archive prior admissions when rebasing onto a newly curated frozen pool",
    )
    args = parser.parse_args()

    league_path = args.league.resolve()
    state_path = args.state.resolve()
    source_pool_path = args.source_base_pool.resolve()
    output_pool_path = args.output_base_pool.resolve()
    reports = find_reports([path.resolve() for path in args.reports_root])
    tactical_evidence = (
        read_json(args.tactical_evidence.resolve())
        if args.tactical_evidence and args.tactical_evidence.is_file()
        else None
    )
    chains = tuple(args.chain or DEFAULT_CHAINS)
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")

    with state_lock(lock_path):
        league = read_json(league_path)
        state = read_json(state_path) if state_path.exists() else {
            "schemaVersion": 1,
            "createdAt": utc_now(),
            "candidates": {},
            "admissions": {},
            "history": [],
        }
        if args.reset_admissions:
            reset_at = utc_now()
            for chain_name, admission in state.get("admissions", {}).items():
                state.setdefault("history", []).append(
                    {**admission, "removedAt": reset_at, "removalReason": "frozen_pool_rebase"}
                )
            state["admissions"] = {}
            state.setdefault("policyChanges", []).append(
                {
                    "at": reset_at,
                    "action": "reset_admissions",
                    "reason": "curated Large-g9 pool keeps only retained best A02/A08 anchors",
                }
            )
        state["policy"] = {
            "chains": list(chains),
            "minFrozenGames": args.min_frozen_games,
            "minDirectGames": args.min_direct_games,
            "maxRegressionPp": args.max_regression_pp,
            "initialMinScore": args.initial_min_score,
            "requiredConsecutivePasses": args.required_passes,
            "engineSeedControlled": False,
            "tacticalEvidence": (
                str(args.tactical_evidence.resolve()) if args.tactical_evidence else None
            ),
        }
        state["sourceBasePool"] = str(source_pool_path)
        state["outputBasePool"] = str(output_pool_path)
        for retired_name, retired_candidate in state.get("candidates", {}).items():
            if retired_name not in chains and retired_candidate.get("status") == "testing":
                retired_candidate["status"] = "superseded"
                retired_candidate["supersededAt"] = utc_now()
        for chain_name in chains:
            if chain_name not in league["chains"]:
                continue
            active = state["candidates"].get(chain_name)
            current_package = select_package(chain_name, league["chains"][chain_name])
            if args.reset_admissions:
                if active:
                    state.setdefault("candidateHistory", []).append(
                        {
                            **active,
                            "supersededAt": utc_now(),
                            "supersededReason": "frozen_pool_rebase",
                        }
                    )
                state["candidates"][chain_name] = new_candidate(current_package)
                state["candidates"][chain_name]["firstAdmission"] = True
                active = state["candidates"][chain_name]
            elif not active:
                state["candidates"][chain_name] = new_candidate(current_package)
                state["candidates"][chain_name]["firstAdmission"] = (
                    chain_name not in state["admissions"]
                )
                active = state["candidates"][chain_name]
            elif active.get("status") in {"admitted", "rejected"}:
                if active.get("snapshotId") == current_package["snapshotId"]:
                    continue
                state["candidates"][chain_name] = new_candidate(current_package)
                state["candidates"][chain_name]["firstAdmission"] = (
                    chain_name not in state["admissions"]
                )
                active = state["candidates"][chain_name]
            active.setdefault("firstAdmission", chain_name not in state["admissions"])
            refresh_candidate_tactical_gate(
                active, current_package.get("tacticalGate", {})
            )
            if not args.register_only:
                update_candidate(
                    active,
                    reports,
                    min_frozen_games=args.min_frozen_games,
                    min_direct_games=args.min_direct_games,
                    max_regression=args.max_regression_pp / 100.0,
                    initial_min_score=args.initial_min_score,
                    required_passes=args.required_passes,
                    tactical_evidence=tactical_evidence,
                )
            if active["status"] == "passed":
                admission = {
                    "admittedAt": utc_now(),
                    "candidate": {key: value for key, value in active.items() if key != "observations"},
                    "agent": frozen_agent(active),
                }
                previous = state["admissions"].get(chain_name)
                if previous:
                    state["history"].append(previous)
                state["admissions"][chain_name] = admission
                active["status"] = "admitted"
                active["admittedAt"] = admission["admittedAt"]

        state["history"] = state["history"][-256:]
        state["updatedAt"] = utc_now()
        state["lastReportsSeen"] = [row["roundId"] for row in reports[-16:]]
        atomic_write_json(state_path, state)

        admissions = {
            key: value for key, value in state["admissions"].items() if key in chains
        }
        admissions["_sourcePool"] = str(source_pool_path)
        source_pool = read_json(source_pool_path)
        frozen_pool = build_frozen_pool(source_pool, admissions)
        ppo_core_path = output_pool_path.with_name(
            f"{output_pool_path.stem}-ppo-core{output_pool_path.suffix}"
        )
        atomic_write_json(ppo_core_path, frozen_pool)
        replace_core_and_reconcile(
            league_path,
            ppo_core_path,
            league_updates={
                "ppoFrozenPromotionState": str(state_path),
                "ppoFrozenPromotionUpdatedAt": utc_now(),
            },
        )

    summary = {
        "updatedAt": state["updatedAt"],
        "candidates": {
            key: {
                "generation": value["generation"],
                "snapshotId": value["snapshotId"],
                "status": value["status"],
                "passRounds": value.get("passRounds", []),
            }
            for key, value in state["candidates"].items()
        },
        "admitted": sorted(state["admissions"]),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
