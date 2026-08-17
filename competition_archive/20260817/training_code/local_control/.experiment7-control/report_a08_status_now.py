#!/usr/bin/env python3
"""Emit a compact current status for Experiment 7 A08 workstreams."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path


CHAIN = "a08_dipplin_seaking"
TARGETS = {
    "archaludon": "public_archaludon_meta",
    "hard_a06": "hard_exploiter_g0010__05_a06_89e6155f2531",
    "diversity_a01": "diversity_g0020__01_a01_ba51a134262b",
}


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def pid_status(path: Path) -> dict:
    try:
        pid = int(path.read_text().strip())
        os.kill(pid, 0)
        return {"pid": pid, "aliveOnOrchestrator": True}
    except Exception as exc:
        return {"pidFile": str(path), "aliveOnOrchestrator": False, "error": str(exc)}


def compact_epoch(metrics: dict) -> dict:
    epochs = metrics.get("epochs") or []
    epoch = epochs[-1] if epochs else {}
    rollouts = metrics.get("rollouts") or []
    return {
        "generation": metrics.get("generation"),
        "createdAt": metrics.get("createdAt"),
        "episodes": sum(int(row.get("episodes", 0)) for row in rollouts),
        "decisions": sum(int(row.get("decisions", 0)) for row in rollouts),
        "kl": epoch.get("approximateKl"),
        "clip": epoch.get("clipFraction"),
        "entropy": epoch.get("entropy"),
        "teacherAnchor": epoch.get("teacherAnchor"),
        "seat1Weight": epoch.get("seat1Weight"),
        "stoppedForKl": metrics.get("stoppedForKl"),
    }


def main() -> None:
    root = Path(sys.argv[1])
    python = sys.argv[2]
    summarizer = sys.argv[3]
    raw = subprocess.check_output(
        [python, "-s", summarizer, "--league-root", str(root)], text=True
    )
    training = json.loads(raw)
    row = training["chains"][CHAIN]
    latest_epoch = row.get("latestEpoch") or {}
    latest_shift = row.get("latestInitialPolicyShift") or {}
    external_games = int(row.get("externalWins", 0)) + int(row.get("externalLosses", 0))

    full = load(root / "monitoring/full-matrix/latest.json")
    matrix = full.get("chains", {}).get(CHAIN, {})
    agents = {entry["agent"]: entry for entry in matrix.get("agents", [])}
    league = load(root / "state/league.json")
    control = league.get("chains", {}).get(CHAIN, {}).get("trainingControl")

    targeted_log = root / "logs/worker-a08-targeted-doraemon17.log"
    targeted_lines = targeted_log.read_text(errors="replace").splitlines()
    targeted_done = [line for line in targeted_lines if line.startswith("SHARD_DONE")]
    targeted_failed = [line for line in targeted_done if "exit=0" not in line]
    targeted_summaries = []
    targeted_opponents: Counter[str] = Counter()
    for path in (root / "buffer/ready" / CHAIN).glob(
        "a08-targeted-*.jsonl.gz.summary.json"
    ):
        summary = load(path)
        targeted_summaries.append(summary)
        targeted_opponents.update(
            {str(name): int(value) for name, value in summary.get("opponents", {}).items()}
        )
    targeted_failed_files = list(
        (root / "buffer/ready" / CHAIN).glob("a08-targeted-*.failed.json")
    )

    learner_receipts = []
    learner_root = root / "learners" / CHAIN
    for generation_dir in sorted(learner_root.glob("generation-*"))[-16:]:
        if (generation_dir / "PUBLISHED.json").is_file():
            state = "published"
        elif (generation_dir / "FAILED.json").is_file():
            state = "failed"
        else:
            state = "incomplete"
        learner_receipts.append({"generationDir": generation_dir.name, "state": state})

    challenger_root = Path(
        "/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-checkpoint-challengers-20260812"
    )
    challengers = {}
    for candidate in sorted(challenger_root.glob("g*/monitoring/full-matrix/latest.json")):
        data = load(candidate)
        result = data.get("chains", {}).get(CHAIN, {})
        candidate_agents = {
            entry["agent"]: entry for entry in result.get("agents", [])
        }
        challengers[candidate.parents[2].name] = {
            "status": data.get("status"),
            "games": data.get("games"),
            "frozen": (result.get("frozenAggregate") or {}).get("scoreRate"),
            "bcFrozen": (result.get("universalBcFrozenAggregate") or {}).get(
                "scoreRate"
            ),
            "ppoMinusBc": result.get("ppoMinusBc"),
            "seat0": (result.get("seatMetrics", {}).get("0") or {}).get("scoreRate"),
            "seat1": (result.get("seatMetrics", {}).get("1") or {}).get("scoreRate"),
            "directBC": (result.get("directVsUniversalBc") or {}).get("scoreRate"),
            "targets": {
                label: (
                    candidate_agents.get(agent, {}).get("ppo") or {}
                ).get("scoreRate")
                for label, agent in TARGETS.items()
            },
        }

    branch_root = Path(
        "/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-deck-branches-20260812"
    )
    branches = {}
    for name in ("a08_maxbelt", "a08_lilligant", "a08_lilligant_maxbelt"):
        metrics = sorted((branch_root / name).glob("generation-*/metrics.json"))
        latest = load(metrics[-1]) if metrics else {}
        branches[name] = {
            **compact_epoch(latest),
            "generationCount": len(metrics),
            "readyForEvalCount": len(list((branch_root / name).glob("generation-*/READY_FOR_EVAL"))),
            "worker": pid_status(branch_root / "workers" / f"{name}.pid"),
        }

    output = {
        "trainingUpdatedAt": training.get("leagueUpdatedAt"),
        "main": {
            "generation": row.get("generation"),
            "completedShards": row.get("completedShards"),
            "episodes": row.get("episodes"),
            "decisions": row.get("decisions"),
            "externalWins": row.get("externalWins"),
            "externalLosses": row.get("externalLosses"),
            "externalWinRate": row.get("externalWins", 0) / external_games,
            "selfPlayEpisodes": row.get("selfPlayEpisodes"),
            "livePpoOpponentEpisodes": row.get("livePpoOpponentEpisodes"),
            "publishedUpdates": row.get("publishedUpdates"),
            "failedUpdates": row.get("failedUpdates"),
            "latestEpoch": {
                key: latest_epoch.get(key)
                for key in (
                    "decisions",
                    "approximateKl",
                    "clipFraction",
                    "entropy",
                    "policyLoss",
                    "valueLoss",
                    "seat1Weight",
                    "teacherAnchor",
                )
            },
            "latestInitialPolicyShift": {
                key: latest_shift.get(key)
                for key in ("decisions", "approximateKl", "clipFraction")
            },
            "trainingControl": control,
        },
        "latestMatrix": {
            "updatedAt": full.get("updatedAt"),
            "generation": matrix.get("generation"),
            "frozen": matrix.get("frozenAggregate"),
            "bcFrozen": matrix.get("universalBcFrozenAggregate"),
            "ppoMinusBc": matrix.get("ppoMinusBc"),
            "delta": matrix.get("deltaVsPrevious"),
            "progress": matrix.get("progress"),
            "seat0": (matrix.get("seatMetrics", {}).get("0") or {}).get("scoreRate"),
            "seat1": (matrix.get("seatMetrics", {}).get("1") or {}).get("scoreRate"),
            "seatGap": matrix.get("seatGap"),
            "directBC": matrix.get("directVsUniversalBc"),
            "targets": {
                label: (agents.get(agent, {}).get("ppo") or {})
                for label, agent in TARGETS.items()
            },
            "headToHead": matrix.get("ppoHeadToHead"),
        },
        "targetedWorker": {
            "receipt": pid_status(root / "workers/a08-targeted-doraemon17.pid"),
            "successfulShards": len(targeted_done) - len(targeted_failed),
            "failedShards": len(targeted_failed),
            "summaryFiles": len(targeted_summaries),
            "summaryFailures": len(targeted_failed_files),
            "episodes": sum(int(row.get("episodes", 0)) for row in targeted_summaries),
            "decisions": sum(int(row.get("decisions", 0)) for row in targeted_summaries),
            "wins": sum(int(row.get("wins", 0)) for row in targeted_summaries),
            "losses": sum(int(row.get("losses", 0)) for row in targeted_summaries),
            "opponents": dict(targeted_opponents),
            "lastLines": targeted_lines[-8:],
        },
        "recentLearnerReceipts": learner_receipts,
        "challengers": challengers,
        "branches": branches,
    }
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
