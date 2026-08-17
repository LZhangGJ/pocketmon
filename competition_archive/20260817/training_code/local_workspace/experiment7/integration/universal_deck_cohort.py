from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_pool(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected", []) if isinstance(payload, dict) else []
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"deck pool is empty: {path}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in selected:
        tier = str(row.get("evidenceTier", "")).upper()
        deck_sha = str(row.get("deckSha256", ""))
        weight = float(row.get("samplingWeight", 0.0))
        if tier not in {"A", "B", "C", "D"}:
            raise ValueError(f"invalid evidence tier: {row}")
        if not deck_sha or deck_sha in seen:
            raise ValueError(f"missing or duplicate deck SHA: {deck_sha}")
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"invalid sampling weight: {row}")
        seen.add(deck_sha)
        rows.append(dict(row))
    return rows


def deterministic_seed(chain_name: str, generation: int, snapshot_sha: str) -> int:
    digest = hashlib.sha256(
        f"{chain_name}:{generation}:{snapshot_sha}:tiered-cohort-v1".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _weighted_without_replacement(
    rows: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    if count > len(rows):
        raise ValueError(f"cannot sample {count} rows from {len(rows)}")
    keyed = []
    for row in rows:
        weight = float(row["samplingWeight"])
        # Efraimidis-Spirakis weighted reservoir sampling.  log(U)/w is
        # numerically stable and the largest keys are selected.
        key = math.log(max(rng.random(), 1e-300)) / weight
        keyed.append((key, str(row["deckSha256"]), row))
    keyed.sort(reverse=True)
    return [dict(row) for _, _, row in keyed[:count]]


def select_cohort(
    rows: list[dict[str, Any]], *, size: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if size <= 0 or size > len(rows):
        raise ValueError(f"invalid cohort size {size} for {len(rows)} decks")
    rng = random.Random(seed)
    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tier_weight: dict[str, float] = defaultdict(float)
    for row in rows:
        tier = str(row["evidenceTier"]).upper()
        by_tier[tier].append(row)
        tier_weight[tier] += float(row["samplingWeight"])
    total_weight = sum(tier_weight.values())
    tier_prob = {tier: tier_weight[tier] / total_weight for tier in "ABCD"}

    # Use floor quotas plus randomized largest-remainder allocation.  This
    # keeps every 20-deck generation close to the configured A/B/C/D mix
    # while preserving stochastic rotation between generations.
    expected = {tier: tier_prob[tier] * size for tier in "ABCD"}
    requested = Counter({tier: math.floor(expected[tier]) for tier in "ABCD"})
    remaining = size - sum(requested.values())
    while remaining:
        eligible = [
            tier
            for tier in "ABCD"
            if requested[tier] < len(by_tier[tier])
        ]
        residual = [max(expected[tier] - requested[tier], 0.0) for tier in eligible]
        if not any(residual):
            residual = [tier_prob[tier] for tier in eligible]
        tier = rng.choices(eligible, weights=residual, k=1)[0]
        requested[tier] += 1
        remaining -= 1
    overflow = 0
    for tier in "ABCD":
        available = len(by_tier[tier])
        if requested[tier] > available:
            overflow += requested[tier] - available
            requested[tier] = available
    while overflow:
        eligible = [tier for tier in "ABCD" if requested[tier] < len(by_tier[tier])]
        tier = rng.choices(eligible, weights=[tier_prob[t] for t in eligible], k=1)[0]
        requested[tier] += 1
        overflow -= 1

    selected: list[dict[str, Any]] = []
    for tier in "ABCD":
        tier_rows = _weighted_without_replacement(by_tier[tier], requested[tier], rng)
        selected_weight = sum(float(row["samplingWeight"]) for row in tier_rows)
        for row in tier_rows:
            # The selected cohort preserves the original rank probabilities.
            row["samplingWeight"] = (
                tier_prob[tier] * float(row["samplingWeight"]) / selected_weight
            )
            selected.append(row)
    selected.sort(key=lambda row: (str(row["evidenceTier"]), str(row["deckSha256"])))
    receipt = {
        "schemaVersion": 1,
        "kind": "universal-ppo-generation-deck-cohort",
        "seed": seed,
        "size": size,
        "tierProbabilities": tier_prob,
        "tierCounts": dict(sorted(Counter(row["evidenceTier"] for row in selected).items())),
        "deckSha256": [str(row["deckSha256"]) for row in selected],
    }
    return selected, receipt


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def materialize_cohort(
    pool_path: Path,
    output_path: Path,
    *,
    size: int,
    chain_name: str,
    generation: int,
    snapshot_sha: str,
) -> dict[str, Any]:
    seed = deterministic_seed(chain_name, generation, snapshot_sha)
    selected, receipt = select_cohort(read_pool(pool_path), size=size, seed=seed)
    payload = {
        **receipt,
        "sourcePool": str(pool_path.resolve()),
        "chain": chain_name,
        "generation": generation,
        "snapshotSha256": snapshot_sha,
        "selected": selected,
    }
    atomic_write_json(output_path, payload)
    return payload
