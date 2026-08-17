from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def score(value: Any) -> float:
    return float(value) if value is not None else -1.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select two complementary PPO Arena candidates from a complete payoff matrix"
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--learners", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary_path = args.summary.resolve()
    learners_path = args.learners.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite selection: {output}")
    summary = load_json(summary_path)
    learners_payload = load_json(learners_path)

    metadata = {row["name"]: row for row in learners_payload.get("agents", [])}
    candidates = []
    for row in summary.get("learners", []):
        name = row["learner"]
        if name not in metadata:
            raise ValueError(f"summary learner is missing metadata: {name}")
        candidates.append({**row, "metadata": metadata[name]})
    if len(candidates) < 2:
        raise ValueError("at least two candidates are required")

    matchups: dict[str, dict[str, dict[str, Any]]] = {}
    opponent_tiers: dict[str, str] = {}
    for row in summary.get("matchups", []):
        matchups.setdefault(row["learner"], {})[row["opponent"]] = row
        opponent_tiers[row["opponent"]] = row.get("skillTier", "unknown")
    expected_opponents = set(opponent_tiers)
    if not expected_opponents:
        raise ValueError("summary contains no matchups")
    for candidate in candidates:
        if set(matchups.get(candidate["learner"], {})) != expected_opponents:
            raise ValueError(f"incomplete payoff row: {candidate['learner']}")

    individual_order = sorted(
        candidates,
        key=lambda row: (
            row.get("failures", 0) == 0,
            score(row.get("hardTierAgentMacroScoreRate")),
            score(row.get("worstOpponentScoreRate")),
            score(row.get("agentMacroScoreRate")),
            score(row.get("submission4ScoreRate")),
            score(row.get("scoreRate")),
        ),
        reverse=True,
    )
    first = individual_order[0]

    pair_rows = []
    for left, right in itertools.combinations(candidates, 2):
        left_meta = left["metadata"]
        right_meta = right["metadata"]
        if left_meta.get("deckSha256") == right_meta.get("deckSha256"):
            continue
        coverage = {
            opponent: max(
                float(matchups[left["learner"]][opponent]["scoreRate"]),
                float(matchups[right["learner"]][opponent]["scoreRate"]),
            )
            for opponent in sorted(expected_opponents)
        }
        hard_values = [
            value for opponent, value in coverage.items() if opponent_tiers[opponent] == "hard"
        ]
        first_weak = [
            opponent
            for opponent in sorted(expected_opponents)
            if float(matchups[first["learner"]][opponent]["scoreRate"]) < 0.5
        ]
        first_weak_lift = sum(
            max(
                0.0,
                coverage[opponent]
                - float(matchups[first["learner"]][opponent]["scoreRate"]),
            )
            for opponent in first_weak
        )
        pair_rows.append(
            {
                "candidates": [left["learner"], right["learner"]],
                "decks": [left_meta.get("deckSha256"), right_meta.get("deckSha256")],
                "zeroFailures": left.get("failures", 0) == 0 and right.get("failures", 0) == 0,
                "coverageWorstOpponentScoreRate": min(coverage.values()),
                "coverageHardTierAgentMacroScoreRate": statistics.mean(hard_values),
                "coverageAgentMacroScoreRate": statistics.mean(coverage.values()),
                "firstChoiceWeakMatchupLift": first_weak_lift,
                "coverage": coverage,
            }
        )
    if not pair_rows:
        raise ValueError("no distinct-deck pair is available")
    pair_rows.sort(
        key=lambda row: (
            row["zeroFailures"],
            row["coverageWorstOpponentScoreRate"],
            row["coverageHardTierAgentMacroScoreRate"],
            row["coverageAgentMacroScoreRate"],
            row["firstChoiceWeakMatchupLift"],
        ),
        reverse=True,
    )
    first_pairs = [row for row in pair_rows if first["learner"] in row["candidates"]]
    if not first_pairs:
        raise ValueError("first choice has no distinct-deck complement")
    complement_pair = first_pairs[0]
    second_name = next(name for name in complement_pair["candidates"] if name != first["learner"])

    role_deltas = []
    by_role: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_role.setdefault(candidate["metadata"]["experiment7Role"], []).append(candidate)
    metric_names = (
        "hardTierAgentMacroScoreRate",
        "worstOpponentScoreRate",
        "agentMacroScoreRate",
        "submission4ScoreRate",
        "scoreRate",
    )
    for role, rows in sorted(by_role.items()):
        baseline_rows = [row for row in rows if row["metadata"]["experiment7Generation"] == 0]
        if not baseline_rows:
            # Narrow screening/confirmation manifests intentionally contain
            # only the promoted candidate for each role.  Role deltas are an
            # audit convenience, not a prerequisite for pair selection.
            continue
        if len(baseline_rows) != 1:
            raise ValueError(f"role must have exactly one g0 baseline: {role}")
        baseline = baseline_rows[0]
        for row in sorted(rows, key=lambda item: item["metadata"]["experiment7Generation"]):
            role_deltas.append(
                {
                    "role": role,
                    "generation": row["metadata"]["experiment7Generation"],
                    "learner": row["learner"],
                    "deltasVsG0": {
                        metric: score(row.get(metric)) - score(baseline.get(metric))
                        for metric in metric_names
                    },
                }
            )

    result = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
            "learners": {"path": str(learners_path), "sha256": sha256_file(learners_path)},
        },
        "decisionRule": {
            "first": "zero failures, then hard-tier macro, worst-opponent, agent macro, submission4, overall",
            "second": "distinct deck maximizing pair worst-opponent coverage, then hard-tier and overall coverage",
            "rolloutWinLossUsed": False,
        },
        "firstChoice": first["learner"],
        "secondChoice": second_name,
        "recommendedPair": complement_pair,
        "bestPairOverall": pair_rows[0],
        "individualOrder": [row["learner"] for row in individual_order],
        "roleDeltasVsG0": role_deltas,
        "pairRanking": pair_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({
        "firstChoice": result["firstChoice"],
        "secondChoice": result["secondChoice"],
        "bestPairOverall": result["bestPairOverall"]["candidates"],
        "output": str(output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
