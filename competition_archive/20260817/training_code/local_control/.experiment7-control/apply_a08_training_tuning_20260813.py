#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


CHAIN = "a08_dipplin_seaking"
WEIGHT_FLOORS = {
    "public_archaludon_meta": 1.60,
    "hard_exploiter_g0010__05_a06_89e6155f2531": 1.60,
    "diversity_g0020__01_a01_ba51a134262b": 1.60,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


@contextmanager
def lock(path: Path):
    import fcntl

    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.league_root.resolve()
    league_path = root / "state/league.json"
    applied_at = now()
    with lock(league_path.with_suffix(league_path.suffix + ".lock")):
        league = read(league_path)
        control = league["chains"][CHAIN]["trainingControl"]
        rollout = control["rollout"]
        learner = control["learner"]
        agent_weights = rollout.setdefault("agentWeights", {})
        for name, floor in WEIGHT_FLOORS.items():
            agent_weights[name] = max(float(agent_weights.get(name, 1.0)), floor)
        rollout.update(
            {
                "selfPlayFraction": 0.20,
                "learnerSeat1Fraction": 0.50,
                "weightPolicy": "bounded latest-cell nudge; A08 high-sample 40-game floors override weak cells",
            }
        )
        learner.update(
            {
                "minDecisions": 12000,
                "maxBehaviorLag": 2,
                "teacherAnchorCoefficient": 0.05,
                "seat1Weight": 1.25,
                "learningRate": 7.5e-6,
                "ppoEpochs": 2,
                "normalizeAdvantagesByPlayer": True,
                "balancePlayerMinibatches": True,
            }
        )
        control["updatedAt"] = applied_at
        control["manualTuning"] = {
            "appliedAt": applied_at,
            "reason": "A08 high-sample weak-matchup specialization with balanced seats",
            "sourceEvidence": "40 games per frozen agent and key opponent",
        }
        league["trainingControlUpdatedAt"] = applied_at
        write(league_path, league)

    receipt = root / "monitoring/adaptive-training/a08-tuning-20260813.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": 1,
        "status": "applied",
        "appliedAt": applied_at,
        "chain": CHAIN,
        "rollout": rollout,
        "learner": learner,
        "takesEffect": "next rollout shard and next learner batch; no main-process restart",
    }
    write(receipt, payload)
    print(json.dumps({**payload, "receipt": str(receipt)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
