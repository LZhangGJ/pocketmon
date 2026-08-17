from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_RESULTS = {"win", "loss", "draw"}
SUBMISSION4 = "team_submission_4_portable_bc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def score(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    points = sum(1.0 if row["result"] == "win" else 0.5 if row["result"] == "draw" else 0.0 for row in rows)
    return points / len(rows)


def wilson_lower(rate: float, games: int, z: float = 1.96) -> float:
    if games <= 0:
        return 0.0
    denominator = 1.0 + z * z / games
    center = rate + z * z / (2.0 * games)
    radius = z * math.sqrt(rate * (1.0 - rate) / games + z * z / (4.0 * games * games))
    return (center - radius) / denominator


def load_view(
    root: Path, manifest: Path | None = None
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]], list[Path]]:
    manifest_path = manifest.resolve() if manifest is not None else root / "learners.json"
    payload = read_json(manifest_path)
    learners = {row["name"]: row for row in payload["agents"]}
    combined = root / "results.csv"
    result_paths = [combined] if combined.is_file() else sorted(root.glob("results-shard-*.csv"))
    if not result_paths or (not combined.is_file() and len(result_paths) != 12):
        raise ValueError(f"expected results.csv or 12 result shards under {root}")
    rows = [row for path in result_paths for row in read_csv(path)]
    return learners, rows, [manifest_path, *result_paths]


def summarize_view(
    learners: dict[str, dict[str, Any]], rows: list[dict[str, str]], games_per_pair: int
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["learner"] not in learners:
            raise ValueError(f"unknown learner in results: {row['learner']}")
        grouped[row["learner"]].append(row)
    summary: dict[str, dict[str, Any]] = {}
    for name, learner in learners.items():
        learner_rows = grouped[name]
        failures = [
            row
            for row in learner_rows
            if row.get("failure", "").strip() or row.get("result", "") not in VALID_RESULTS
        ]
        valid = [row for row in learner_rows if row not in failures]
        opponents = {
            opponent: [row for row in valid if row["opponent"] == opponent]
            for opponent in sorted({row["opponent"] for row in learner_rows})
        }
        if len(learner_rows) != 4 * games_per_pair or len(opponents) != 4:
            raise ValueError(f"unexpected Arena coverage for {name}: games={len(learner_rows)} opponents={len(opponents)}")
        if any(len(group) != games_per_pair for group in opponents.values()):
            raise ValueError(f"unexpected per-opponent coverage for {name}")
        seat_rates = []
        for seat in ("0", "1"):
            seat_rows = [row for row in valid if row["learner_seat"] == seat]
            seat_rates.append(score(seat_rows))
        summary[str(learner["deckSha256"])] = {
            "learner": name,
            "games": len(learner_rows),
            "failures": len(failures),
            "scoreRate": score(valid),
            "opponents": {opponent: score(group) for opponent, group in opponents.items()},
            "seatGap": abs(seat_rates[0] - seat_rates[1]),
        }
    return summary


def ranking_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -row["failures"],
        row["minPolicyScoreRate"],
        row["combinedScoreRate"],
        row["worstOpponentCombinedScoreRate"],
        row["submission4CombinedScoreRate"],
        -row["maxPolicySeatGap"],
        row["deckSha256"],
    )


def write_manifest(path: Path, source: dict[str, dict[str, Any]], selected: list[dict[str, Any]]) -> None:
    by_sha = {str(row["deckSha256"]): row for row in source.values()}
    agents = [by_sha[row["deckSha256"]] for row in selected]
    payload = {"schemaVersion": 1, "agents": agents}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select robust decks from two Arena policy views")
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--g0-root", type=Path, required=True)
    parser.add_argument("--g20-root", type=Path, required=True)
    parser.add_argument("--g0-learners", type=Path)
    parser.add_argument("--g20-learners", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--games-per-pair", type=int, default=20)
    parser.add_argument("--coverage-mode", choices=("diverse", "global"), default="diverse")
    args = parser.parse_args()

    population_path = args.population.resolve()
    g0_root = args.g0_root.resolve()
    g20_root = args.g20_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    population = read_json(population_path)
    metadata = {str(row["deckSha256"]): row for row in population["selected"]}
    g0_learners, g0_rows, g0_sources = load_view(g0_root, args.g0_learners)
    g20_learners, g20_rows, g20_sources = load_view(g20_root, args.g20_learners)
    g0 = summarize_view(g0_learners, g0_rows, args.games_per_pair)
    g20 = summarize_view(g20_learners, g20_rows, args.games_per_pair)
    if set(g0) != set(g20):
        raise ValueError("two policy views do not contain the same deck hashes")
    missing_metadata = set(g0) - set(metadata)
    if missing_metadata:
        raise ValueError(f"population metadata are missing deck hashes: {sorted(missing_metadata)}")

    ranking: list[dict[str, Any]] = []
    for deck_sha in sorted(g0):
        meta = metadata[deck_sha]
        views = {"g0": g0[deck_sha], "diversity_g20": g20[deck_sha]}
        opponent_names = sorted(g0[deck_sha]["opponents"])
        combined_opponents = {
            opponent: (g0[deck_sha]["opponents"][opponent] + g20[deck_sha]["opponents"][opponent]) / 2.0
            for opponent in opponent_names
        }
        combined = (g0[deck_sha]["scoreRate"] + g20[deck_sha]["scoreRate"]) / 2.0
        worst_opponent = min(combined_opponents, key=lambda name: (combined_opponents[name], name))
        row = {
            "candidate": meta["name"],
            "deckSha256": deck_sha,
            "deckPath": meta["deckPath"],
            "archetypeId": meta["archetypeId"],
            "archetypeLabel": meta["archetypeLabel"],
            "method": meta["method"],
            "parents": list(meta["parents"]),
            "failures": g0[deck_sha]["failures"] + g20[deck_sha]["failures"],
            "combinedGames": g0[deck_sha]["games"] + g20[deck_sha]["games"],
            "combinedScoreRate": combined,
            "combinedWilson95Lower": wilson_lower(combined, 8 * args.games_per_pair),
            "minPolicyScoreRate": min(g0[deck_sha]["scoreRate"], g20[deck_sha]["scoreRate"]),
            "worstOpponent": worst_opponent,
            "worstOpponentCombinedScoreRate": combined_opponents[worst_opponent],
            "submission4CombinedScoreRate": combined_opponents[SUBMISSION4],
            "maxPolicySeatGap": max(g0[deck_sha]["seatGap"], g20[deck_sha]["seatGap"]),
            "policyViews": views,
            "combinedOpponents": combined_opponents,
        }
        ranking.append(row)
    ranking.sort(key=ranking_key, reverse=True)
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index

    if any(row["failures"] for row in ranking):
        raise ValueError("round20 deck search contains failures")
    selected: list[dict[str, Any]] = []
    reasons: dict[str, list[str]] = defaultdict(list)

    def choose(row: dict[str, Any], reason: str) -> None:
        if row not in selected and len(selected) < args.top_k:
            selected.append(row)
        reasons[row["deckSha256"]].append(reason)

    if args.coverage_mode == "diverse":
        unique_bases = []
        seen_base_decks = set()
        for base in population["bases"]:
            if base["deckSha256"] in seen_base_decks or str(base["name"]).lower() == "submission4":
                continue
            seen_base_decks.add(base["deckSha256"])
            unique_bases.append(str(base["name"]))
        for parent in unique_bases:
            candidates = [row for row in ranking if parent in row["parents"]]
            if candidates:
                choose(candidates[0], f"best_parent:{parent}")
        for method in sorted(population["methodCounts"]):
            candidates = [row for row in ranking if row["method"] == method]
            if candidates:
                choose(candidates[0], f"best_method:{method}")
    for row in ranking:
        if len(selected) >= args.top_k:
            break
        choose(row, "global_rank_fill")
    selected.sort(key=lambda row: row["rank"])

    ranking_csv = output / "ranking.csv"
    fields = [
        "rank", "candidate", "deckSha256", "archetypeId", "method", "parents", "failures",
        "combinedGames", "combinedScoreRate", "combinedWilson95Lower", "minPolicyScoreRate",
        "worstOpponent", "worstOpponentCombinedScoreRate", "submission4CombinedScoreRate",
        "maxPolicySeatGap", "g0ScoreRate", "diversityG20ScoreRate", "selected", "selectionReasons",
    ]
    with ranking_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in ranking:
            writer.writerow({
                "rank": row["rank"],
                "candidate": row["candidate"],
                "deckSha256": row["deckSha256"],
                "archetypeId": row["archetypeId"],
                "method": row["method"],
                "parents": "+".join(row["parents"]),
                "failures": row["failures"],
                "combinedGames": row["combinedGames"],
                "combinedScoreRate": row["combinedScoreRate"],
                "combinedWilson95Lower": row["combinedWilson95Lower"],
                "minPolicyScoreRate": row["minPolicyScoreRate"],
                "worstOpponent": row["worstOpponent"],
                "worstOpponentCombinedScoreRate": row["worstOpponentCombinedScoreRate"],
                "submission4CombinedScoreRate": row["submission4CombinedScoreRate"],
                "maxPolicySeatGap": row["maxPolicySeatGap"],
                "g0ScoreRate": row["policyViews"]["g0"]["scoreRate"],
                "diversityG20ScoreRate": row["policyViews"]["diversity_g20"]["scoreRate"],
                "selected": row in selected,
                "selectionReasons": ";".join(reasons.get(row["deckSha256"], [])),
            })

    g0_manifest = output / f"top{args.top_k}_learners_g0.json"
    g20_manifest = output / f"top{args.top_k}_learners_diversity_g20.json"
    write_manifest(g0_manifest, g0_learners, selected)
    write_manifest(g20_manifest, g20_learners, selected)
    selection = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "decisionRule": {
            "ranking": "zero failures, min policy score, combined score, worst combined opponent, submission4, seat gap",
            "coverage": (
                "best candidate involving each unique base parent and each generation method, then global rank fill"
                if args.coverage_mode == "diverse"
                else "global rank only"
            ),
            "topK": args.top_k,
            "gamesPerPairPerPolicyView": args.games_per_pair,
        },
        "sources": {
            "population": {"path": str(population_path), "sha256": sha256_file(population_path)},
            "g0": [{"path": str(path), "sha256": sha256_file(path)} for path in g0_sources],
            "diversityG20": [{"path": str(path), "sha256": sha256_file(path)} for path in g20_sources],
        },
        "selected": [
            {
                **{key: row[key] for key in (
                    "rank", "candidate", "deckSha256", "deckPath", "archetypeId", "method", "parents",
                    "combinedScoreRate", "minPolicyScoreRate", "worstOpponent",
                    "worstOpponentCombinedScoreRate", "submission4CombinedScoreRate",
                )},
                "selectionReasons": reasons[row["deckSha256"]],
            }
            for row in selected
        ],
    }
    selection_path = output / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "SHA256SUMS").open("x", encoding="utf-8") as handle:
        for path in (ranking_csv, selection_path, g0_manifest, g20_manifest):
            handle.write(f"{sha256_file(path)}  {path}\n")
    (output / "SUCCESS").touch()
    print(json.dumps({"output": str(output), "selected": len(selected), "top": selected[0]["candidate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
