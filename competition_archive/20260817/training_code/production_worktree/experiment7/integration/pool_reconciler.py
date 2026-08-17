from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

try:
    from async_ppo_control import (
        atomic_write_json,
        build_pool_payload,
        read_json,
        sha256_file,
        state_lock,
    )
except ModuleNotFoundError:
    from .async_ppo_control import (
        atomic_write_json,
        build_pool_payload,
        read_json,
        sha256_file,
        state_lock,
    )


RECONCILER_SCHEMA_VERSION = 1
RECONCILER_KEY = "canonicalPoolReconciler"


def policy_key(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("deck_canonical_sha256") or row.get("deckSha256") or ""),
        str(row.get("policyVersion") or row.get("behavior_checkpoint_sha256") or ""),
    )


def _eligible(row: dict[str, Any], source_kind: str | None) -> bool:
    if source_kind is None:
        return True
    return (
        row.get("sourceKind") == source_kind
        and row.get("immutable") is True
        and row.get("ppoUpdatesAllowed") is False
    )


def _source_record(path: Path, source_kind: str | None) -> dict[str, Any]:
    payload = read_json(path)
    agents = [row for row in payload.get("agents", []) if _eligible(row, source_kind)]
    if not agents:
        raise ValueError(f"canonical pool source has no admissible agents: {path}")
    if source_kind is not None and any(not all(policy_key(row)) for row in agents):
        raise ValueError(f"canonical source agent lacks deck/policy identity: {path}")
    keys = [policy_key(row) for row in agents if all(policy_key(row))]
    if source_kind is not None and len(keys) != len(set(keys)):
        raise ValueError(f"canonical pool source contains duplicate policy keys: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "sourceKind": source_kind,
        "agents": len(agents),
    }


def _write_core_without_managed_sources(
    base_path: Path,
    core_path: Path,
    managed_source_kinds: set[str],
) -> None:
    current = read_json(base_path)
    core = {
        key: value
        for key, value in current.items()
        if key
        not in {
            "agents",
            "canonicalPoolReconciler",
            "latestFrozenPolicyPool",
            "staticFrozenReplayBcPool",
        }
    }
    core["agents"] = [
        row
        for row in current.get("agents", [])
        if str(row.get("sourceKind") or "") not in managed_source_kinds
    ]
    if not core["agents"]:
        raise ValueError("cannot initialize an empty canonical core pool")
    atomic_write_json(core_path, core)


def _configured_sources(league: dict[str, Any]) -> list[dict[str, Any]]:
    config = league.get(RECONCILER_KEY, {})
    sources = config.get("sources", [])
    if not isinstance(sources, list):
        raise TypeError("canonical pool reconciler sources must be a list")
    return [dict(row) for row in sources]


def _merge_sources(sources: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    core_sources = [row for row in sources if row.get("role") == "core"]
    if len(core_sources) != 1:
        raise ValueError("canonical pool reconciler requires exactly one core source")
    overlays = sorted(
        (row for row in sources if row.get("role") == "overlay"),
        key=lambda row: (str(row.get("sourceKind", "")), str(row.get("path", ""))),
    )
    ordered_sources = core_sources + overlays
    records: list[dict[str, Any]] = []
    merged: list[dict[str, Any]] = []
    key_to_index: dict[tuple[str, str], int] = {}
    core_payload: dict[str, Any] | None = None
    for source in ordered_sources:
        path = Path(source["path"]).resolve()
        source_kind = source.get("sourceKind")
        payload = read_json(path)
        if core_payload is None:
            core_payload = payload
        rows = [row for row in payload.get("agents", []) if _eligible(row, source_kind)]
        record = _source_record(path, source_kind)
        record["role"] = source["role"]
        records.append(record)
        for row in rows:
            key = policy_key(row)
            if all(key):
                previous = key_to_index.get(key)
                if previous is not None:
                    merged[previous] = row
                    continue
                key_to_index[key] = len(merged)
            merged.append(row)
    assert core_payload is not None
    result = {
        key: value
        for key, value in core_payload.items()
        if key not in {"agents", RECONCILER_KEY}
    }
    result["agents"] = merged
    result[RECONCILER_KEY] = {
        "schemaVersion": RECONCILER_SCHEMA_VERSION,
        "dedupe": "deck_identity_plus_policy_version",
        "sources": records,
    }
    return result, records


def _reconcile_locked(
    league_path: Path,
    league: dict[str, Any],
    build_live: Callable[[dict[str, Any]], dict[str, Any]] = build_pool_payload,
) -> dict[str, Any]:
    base_path = Path(league["basePool"]["path"]).resolve()
    live_path = Path(league["poolPath"]).resolve()
    previous_base_agents = len(read_json(base_path).get("agents", []))
    prospective_base, records = _merge_sources(_configured_sources(league))
    base_temp = base_path.with_name(f".{base_path.name}.{os.getpid()}.reconciled")
    try:
        base_temp.write_text(
            json.dumps(prospective_base, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        prospective_league = {
            **league,
            "basePool": {**league["basePool"], "path": str(base_temp)},
        }
        live = build_live(prospective_league)
        live.setdefault("asyncLeague", {})["basePool"] = {
            "path": str(base_path),
            "sha256": sha256_file(base_temp),
        }
        atomic_write_json(base_path, prospective_base)
        atomic_write_json(live_path, live)
        league["poolSha256"] = sha256_file(live_path)
        league[RECONCILER_KEY]["lastBaseSha256"] = sha256_file(base_path)
        league[RECONCILER_KEY]["lastLiveSha256"] = league["poolSha256"]
        atomic_write_json(league_path, league)
    finally:
        base_temp.unlink(missing_ok=True)
    return {
        "basePool": {
            "path": str(base_path),
            "before": previous_base_agents,
            "after": len(prospective_base["agents"]),
            "agents": len(prospective_base["agents"]),
            "sha256": sha256_file(base_path),
        },
        "livePool": {
            "path": str(live_path),
            "agents": len(live["agents"]),
            "sha256": sha256_file(live_path),
        },
        "sources": records,
    }


def register_source_and_reconcile(
    league_path: Path,
    source_path: Path,
    source_kind: str,
    *,
    build_live: Callable[[dict[str, Any]], dict[str, Any]] = build_pool_payload,
) -> dict[str, Any]:
    league_path = league_path.resolve()
    source_path = source_path.resolve()
    with state_lock(league_path.with_suffix(league_path.suffix + ".lock")):
        league = read_json(league_path)
        config = dict(league.get(RECONCILER_KEY, {}))
        sources = _configured_sources(league)
        if not sources:
            base_path = Path(league["basePool"]["path"]).resolve()
            core_path = base_path.with_name(f"{base_path.stem}-canonical-core.json")
            managed = {source_kind}
            for key in ("latestStaticFrozenBcAdmission", "latestFrozenPolicyAdmission"):
                row = league.get(key, {})
                if row.get("sourceKind"):
                    managed.add(str(row["sourceKind"]))
                elif key == "latestStaticFrozenBcAdmission" and row.get("pool"):
                    managed.add("replay_static_bc")
            _write_core_without_managed_sources(base_path, core_path, managed)
            sources = [{"role": "core", "path": str(core_path), "sourceKind": None}]
            static_admission = league.get("latestStaticFrozenBcAdmission", {})
            if static_admission.get("pool"):
                sources.append(
                    {
                        "role": "overlay",
                        "path": str(Path(static_admission["pool"]).resolve()),
                        "sourceKind": "replay_static_bc",
                    }
                )
            frozen_admission = league.get("latestFrozenPolicyAdmission", {})
            if frozen_admission.get("pool") and frozen_admission.get("sourceKind"):
                sources.append(
                    {
                        "role": "overlay",
                        "path": str(Path(frozen_admission["pool"]).resolve()),
                        "sourceKind": str(frozen_admission["sourceKind"]),
                    }
                )
        sources = [
            row
            for row in sources
            if not (row.get("role") == "overlay" and row.get("sourceKind") == source_kind)
        ]
        sources.append(
            {
                "role": "overlay",
                "path": str(source_path),
                "sourceKind": source_kind,
            }
        )
        config.update({"schemaVersion": RECONCILER_SCHEMA_VERSION, "sources": sources})
        league[RECONCILER_KEY] = config
        if source_kind == "replay_static_bc":
            league["latestStaticFrozenBcAdmission"] = {
                "pool": str(source_path),
                "agents": _source_record(source_path, source_kind)["agents"],
            }
        else:
            league["latestFrozenPolicyAdmission"] = {
                "pool": str(source_path),
                "sourceKind": source_kind,
                "agents": _source_record(source_path, source_kind)["agents"],
            }
        return _reconcile_locked(league_path, league, build_live=build_live)


def replace_core_and_reconcile(
    league_path: Path,
    core_path: Path,
    *,
    build_live: Callable[[dict[str, Any]], dict[str, Any]] = build_pool_payload,
    league_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    league_path = league_path.resolve()
    core_path = core_path.resolve()
    with state_lock(league_path.with_suffix(league_path.suffix + ".lock")):
        league = read_json(league_path)
        if league_updates:
            league.update(league_updates)
        sources = [row for row in _configured_sources(league) if row.get("role") != "core"]
        sources.insert(0, {"role": "core", "path": str(core_path), "sourceKind": None})
        league[RECONCILER_KEY] = {
            **league.get(RECONCILER_KEY, {}),
            "schemaVersion": RECONCILER_SCHEMA_VERSION,
            "sources": sources,
        }
        return _reconcile_locked(league_path, league, build_live=build_live)
