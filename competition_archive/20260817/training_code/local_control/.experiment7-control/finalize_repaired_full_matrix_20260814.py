from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LEAGUE_ROOT = Path(
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811"
)
EVAL_ROOT = LEAGUE_ROOT / "monitoring" / "full-matrix"
ROUND_ID = (
    "20260814T040813Z-a02_grim_g247-g000286-a02_grim_g247_pokegear-g000286-"
    "a08_maxbelt-g000306-a08_rabsca-g000306-lucario_gold_exact-g000018-"
    "universal_ppo_large_256x6-g000009-universal_ppo_standard_1m-g000011"
)
STAGING = EVAL_ROOT / "rounds" / f".{ROUND_ID}.in-progress"
FINAL = EVAL_ROOT / "rounds" / ROUND_ID
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def metrics(rows: list[dict[str, str]], invert: bool = False) -> dict[str, Any]:
    values = [row.get("result") for row in rows]
    wins = sum(value == ("loss" if invert else "win") for value in values)
    losses = sum(value == ("win" if invert else "loss") for value in values)
    draws = sum(value == "draw" for value in values)
    failures = sum(value in FAILURE_RESULTS for value in values)
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


def key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return row["learner"], row["opponent"], row["seed"], row["learner_seat"]


def main() -> None:
    if not STAGING.is_dir():
        raise FileNotFoundError(STAGING)
    if FINAL.exists():
        raise FileExistsError(FINAL)

    metadata = load(STAGING / "metadata.json")
    if metadata.get("roundId") != ROUND_ID:
        raise ValueError("round id mismatch")
    selected = metadata["selected"]
    opponent_manifest = load(STAGING / "opponents.json")
    opponents_by_name = {row["name"]: row for row in opponent_manifest.get("agents", [])}
    frozen = [opponents_by_name[name] for name in metadata["frozenAgents"]]
    frozen_names = set(metadata["frozenAgents"])

    schedule = read_csv(STAGING / "schedule.csv")
    paths = sorted((STAGING / "raw").glob("results-shard-*.csv"))
    if len(paths) != 15:
        raise RuntimeError(f"expected 15 shard files, found {len(paths)}")
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_csv(path))
    if len(schedule) != 2576 or len(rows) != len(schedule):
        raise RuntimeError(f"coverage mismatch: schedule={len(schedule)} results={len(rows)}")
    if Counter(map(key, schedule)) != Counter(map(key, rows)):
        raise RuntimeError("result keys do not exactly match the schedule")
    all_metrics = metrics(rows)
    if all_metrics["failures"]:
        raise RuntimeError(f"refusing to finalize {all_metrics['failures']} failed games")

    previous_path = EVAL_ROOT / "latest.json"
    previous = load(previous_path) if previous_path.is_file() else None
    previous_compatible = bool(
        previous and previous.get("evaluationDesignVersion") == EVALUATION_DESIGN_VERSION
    )
    report: dict[str, Any] = {}
    for selected_row in selected:
        chain = selected_row["chain"]
        ppo_name = selected_row["learner"]
        bc_name = selected_row["bcLearner"]
        agent_rows = []
        for opponent in frozen:
            ppo_metric = metrics(
                [row for row in rows if row["learner"] == ppo_name and row["opponent"] == opponent["name"]]
            )
            bc_metric = metrics(
                [row for row in rows if row["learner"] == bc_name and row["opponent"] == opponent["name"]]
            )
            previous_agent = None
            if previous_compatible:
                previous_agent = next(
                    (
                        item
                        for item in previous.get("chains", {}).get(chain, {}).get("agents", [])
                        if item["agent"] == opponent["name"]
                    ),
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
                        if previous_agent
                        and ppo_metric["scoreRate"] is not None
                        and previous_agent["ppo"]["scoreRate"] is not None
                        else None
                    ),
                }
            )

        ppo_frozen_rows = [
            row for row in rows if row["learner"] == ppo_name and row["opponent"] in frozen_names
        ]
        bc_frozen_rows = [
            row for row in rows if row["learner"] == bc_name and row["opponent"] in frozen_names
        ]
        aggregate = metrics(ppo_frozen_rows)
        bc_aggregate = metrics(bc_frozen_rows)
        previous_aggregate = (
            previous.get("chains", {}).get(chain, {}).get("frozenAggregate")
            if previous_compatible
            else None
        )
        delta_previous = (
            aggregate["scoreRate"] - previous_aggregate["scoreRate"]
            if previous_aggregate
            and aggregate["scoreRate"] is not None
            and previous_aggregate["scoreRate"] is not None
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
            direct = [
                row for row in rows if row["learner"] == ppo_name and row["opponent"] == other["learner"]
            ]
            inverse = [
                row for row in rows if row["learner"] == other["learner"] and row["opponent"] == ppo_name
            ]
            head_to_head[other["chain"]] = metrics(direct) if direct else metrics(inverse, invert=True)

        seats = {
            seat: metrics([row for row in ppo_frozen_rows if row["learner_seat"] == seat])
            for seat in ("0", "1")
        }
        report[chain] = {
            "generation": selected_row["generation"],
            "snapshotId": selected_row["snapshotId"],
            "frozenAggregate": aggregate,
            "universalBcFrozenAggregate": bc_aggregate,
            "ppoMinusBc": aggregate["scoreRate"] - bc_aggregate["scoreRate"],
            "deltaVsPrevious": delta_previous,
            "progress": progress,
            "seatMetrics": seats,
            "seatGap": abs(seats["0"]["scoreRate"] - seats["1"]["scoreRate"]),
            "directVsUniversalBc": metrics(
                [row for row in rows if row["learner"] == ppo_name and row["opponent"] == bc_name]
            ),
            "ppoHeadToHead": head_to_head,
            "agents": agent_rows,
        }

    completed = now()
    metadata.update({"status": "complete", "completedAt": completed.isoformat(), "games": len(rows)})
    write(STAGING / "metadata.json", metadata)
    payload = {
        "schemaVersion": 1,
        "evaluationDesignVersion": EVALUATION_DESIGN_VERSION,
        "status": "complete",
        "busy": False,
        "updatedAt": completed.isoformat(),
        "roundId": ROUND_ID,
        "games": len(rows),
        "engineSeedControlled": all(
            row.get("engine_seed_controlled", "").lower() == "true" for row in rows
        ),
        "seedPolicy": "fixed paired-seat Python agent RNG seeds; native engine deal RNG is uncontrolled",
        "frozenAgentCount": len(frozen),
        "chains": report,
        "previousRoundId": previous.get("roundId") if previous_compatible else None,
        "recovery": {
            "reason": "two originally assigned hosts lacked /proc/pressure/io",
            "repairedShards": [1, 2, 6, 7, 11, 12],
        },
    }
    write(STAGING / "report.json", payload)
    os.replace(STAGING, FINAL)
    payload["roundPath"] = str(FINAL)
    atomic_write(EVAL_ROOT / "latest.json", payload)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
