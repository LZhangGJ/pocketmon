#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CHAIN_DEFAULTS = {
    "a02_submission4_grimmsnarl_froslass_munkidori": {
        "minDecisions": 6000,
        "seat1Weight": 1.0,
    },
    "a05_raging_bolt_ogerpon_kangaskhan": {
        "minDecisions": 7000,
        "seat1Weight": 1.0,
    },
    "a08_dipplin_seaking": {
        "minDecisions": 12000,
        "seat1Weight": 1.25,
    },
    "mega_lucario_ex": {
        "minDecisions": 8000,
        "seat1Weight": 1.0,
    },
    "lucario_gold_exact": {
        "minDecisions": 8000,
        "seat1Weight": 1.0,
    },
}

A08_HIGH_SAMPLE_AGENT_WEIGHT_FLOORS = {
    "public_archaludon_meta": 1.60,
    "hard_exploiter_g0010__05_a06_89e6155f2531": 1.60,
    "diversity_g0020__01_a01_ba51a134262b": 1.60,
}


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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@contextmanager
def state_lock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rounded(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def smooth(previous: float, proposed: float, maximum_step: float) -> float:
    return previous + clamp(proposed - previous, -maximum_step, maximum_step)


def score_rate(row: dict[str, Any] | None, default: float = 0.5) -> float:
    if not isinstance(row, dict):
        return default
    value = row.get("scoreRate", default)
    return float(value) if isinstance(value, (int, float)) else default


def proposed_seat_controls(gap: float, chain_name: str) -> tuple[float, float]:
    # Positive gap means seat 1 (the second player) is weaker.
    if gap <= 0.04:
        weight = 1.0
    elif gap <= 0.09:
        weight = 1.25
    elif gap <= 0.15:
        weight = 1.5
    else:
        weight = 2.0
    seat1_fraction = 0.5 + clamp(gap * 0.75, -0.10, 0.20)
    return weight, clamp(seat1_fraction, 0.40, 0.70)


def agent_weight(agent: dict[str, Any]) -> float:
    ppo_rate = score_rate(agent.get("ppo"))
    delta = float(agent.get("deltaVsPrevious") or 0.0)
    # A single cell is four games, so it can only create a bounded nudge.
    weight = 1.0 + 1.2 * (0.5 - ppo_rate)
    if delta < 0:
        weight += min(abs(delta), 0.25) * 0.6
    return rounded(clamp(weight, 0.70, 1.60), 2)


def build_control(
    chain_name: str,
    chain: dict[str, Any],
    matrix_row: dict[str, Any],
    previous: dict[str, Any] | None,
    chain_archetypes: dict[str, str],
    round_id: str,
) -> dict[str, Any]:
    previous = previous or {}
    previous_rollout = previous.get("rollout", {})
    previous_learner = previous.get("learner", {})
    defaults = CHAIN_DEFAULTS[chain_name]

    seats = matrix_row.get("seatMetrics", {})
    seat0 = score_rate(seats.get("0"))
    seat1 = score_rate(seats.get("1"))
    gap = seat0 - seat1
    target_seat_weight, target_seat1_fraction = proposed_seat_controls(gap, chain_name)
    old_seat_weight = float(
        previous_learner.get("seat1Weight", defaults["seat1Weight"])
    )
    old_seat_fraction = float(previous_rollout.get("learnerSeat1Fraction", 0.5))
    seat_weight = rounded(smooth(old_seat_weight, target_seat_weight, 0.25), 2)
    seat1_fraction = rounded(
        smooth(old_seat_fraction, target_seat1_fraction, 0.05), 2
    )

    delta = float(matrix_row.get("deltaVsPrevious") or 0.0)
    progress = str(matrix_row.get("progress", "flat"))
    ppo_minus_bc = float(matrix_row.get("ppoMinusBc") or 0.0)
    direct_bc = score_rate(matrix_row.get("directVsUniversalBc"))
    if progress == "regressed" or delta < -0.02:
        target_self_play = 0.15
    elif progress == "flat":
        target_self_play = 0.20
    elif direct_bc >= 0.65:
        target_self_play = 0.25
    else:
        target_self_play = 0.20
    old_self_play = float(previous_rollout.get("selfPlayFraction", 0.25))
    self_play = rounded(smooth(old_self_play, target_self_play, 0.05), 2)

    agent_weights = {
        str(agent["agent"]): agent_weight(agent)
        for agent in matrix_row.get("agents", [])
        if isinstance(agent, dict) and agent.get("agent")
    }
    if chain_name == "a08_dipplin_seaking":
        # These floors come from the completed 40-game-per-agent challenger
        # screen, and therefore take precedence over noisy four-game cells.
        for agent_name, floor in A08_HIGH_SAMPLE_AGENT_WEIGHT_FLOORS.items():
            agent_weights[agent_name] = max(agent_weights.get(agent_name, 1.0), floor)
    archetype_weights: dict[str, float] = {}
    for opponent_chain, head_to_head in matrix_row.get("ppoHeadToHead", {}).items():
        archetype = chain_archetypes.get(opponent_chain)
        if not archetype:
            continue
        rate = score_rate(head_to_head)
        archetype_weights[archetype] = rounded(
            clamp(1.0 + 1.2 * (0.5 - rate), 0.75, 1.50), 2
        )

    min_decisions = int(defaults["minDecisions"])
    if progress == "regressed" or delta < -0.02:
        min_decisions += 2000
    if gap > 0.10:
        min_decisions += 1000
    min_decisions = min(max(min_decisions, 5000), 12000)

    if ppo_minus_bc < 0 or direct_bc < 0.50:
        teacher_anchor = 0.05
    elif progress == "regressed":
        teacher_anchor = 0.04
    elif direct_bc >= 0.65 and ppo_minus_bc >= 0.05:
        teacher_anchor = 0.02
    else:
        teacher_anchor = 0.03
    learning_rate = 5e-6 if progress == "regressed" or delta < -0.02 else 1e-5
    ppo_epochs = 1
    if chain_name == "a08_dipplin_seaking":
        min_decisions = 12000
        teacher_anchor = 0.05
        learning_rate = 7.5e-6
        ppo_epochs = 2

    return {
        "schemaVersion": 1,
        "updatedAt": utc_now(),
        "sourceRoundId": round_id,
        "evidence": {
            "generation": int(matrix_row.get("generation", -1)),
            "frozenScoreRate": score_rate(matrix_row.get("frozenAggregate")),
            "ppoMinusBc": ppo_minus_bc,
            "directVsBcScoreRate": direct_bc,
            "deltaVsPrevious": delta,
            "progress": progress,
            "seat0ScoreRate": seat0,
            "seat1ScoreRate": seat1,
            "seatGap": gap,
            "engineSeedControlled": False,
        },
        "rollout": {
            "selfPlayFraction": self_play,
            "learnerSeat1Fraction": seat1_fraction,
            "archetypeWeights": archetype_weights,
            "agentWeights": agent_weights,
            "weightPolicy": (
                "bounded latest-cell nudge; A08 high-sample 40-game floors override weak cells"
                if chain_name == "a08_dipplin_seaking"
                else "bounded latest-cell nudge; full-matrix aggregate drives learner controls"
            ),
        },
        "learner": {
            "minDecisions": min_decisions,
            "maxBehaviorLag": 2,
            "teacherAnchorCoefficient": teacher_anchor,
            "seat1Weight": seat_weight,
            "learningRate": learning_rate,
            "ppoEpochs": ppo_epochs,
            "normalizeAdvantagesByPlayer": True,
            "balancePlayerMinibatches": True,
        },
    }


def apply_once(league_path: Path, matrix_path: Path, state_path: Path) -> bool:
    matrix = read_json(matrix_path)
    if matrix.get("status") != "complete" or matrix.get("busy"):
        return False
    round_id = str(matrix.get("roundId") or "")
    if not round_id:
        raise ValueError("complete matrix has no roundId")
    state = read_json(state_path) if state_path.exists() else {"schemaVersion": 1}
    if state.get("lastRoundId") == round_id:
        return False

    with state_lock(league_path.with_suffix(league_path.suffix + ".lock")):
        # Re-read under the same lock used by checkpoint publication so a
        # simultaneous generation update cannot be lost.
        league = read_json(league_path)
        chain_archetypes = {
            name: str(row["archetypeId"]).upper()
            for name, row in league["chains"].items()
        }
        receipts = {}
        for chain_name, defaults in CHAIN_DEFAULTS.items():
            if chain_name not in league["chains"]:
                continue
            matrix_row = matrix.get("chains", {}).get(chain_name)
            if not isinstance(matrix_row, dict):
                raise KeyError(f"matrix is missing chain {chain_name}")
            chain = league["chains"][chain_name]
            control = build_control(
                chain_name,
                chain,
                matrix_row,
                chain.get("trainingControl"),
                chain_archetypes,
                round_id,
            )
            chain["trainingControl"] = control
            receipts[chain_name] = control
        league["trainingControlUpdatedAt"] = utc_now()
        league["trainingControlSourceRoundId"] = round_id
        atomic_write_json(league_path, league)
    state.update(
        {
            "lastRoundId": round_id,
            "lastAppliedAt": utc_now(),
            "matrixUpdatedAt": matrix.get("updatedAt"),
            "controls": receipts,
        }
    )
    atomic_write_json(state_path, state)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt all configured PPO chains from complete Arena evidence")
    parser.add_argument("--league", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            changed = apply_once(args.league, args.matrix, args.state)
            print(
                json.dumps(
                    {"at": utc_now(), "changed": changed}, ensure_ascii=False
                ),
                flush=True,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {"at": utc_now(), "error": f"{type(exc).__name__}: {exc}"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.once:
                raise
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
