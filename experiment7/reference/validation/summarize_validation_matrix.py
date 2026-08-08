#!/usr/bin/env python3
"""Validate and summarize a frozen multi-opponent arena matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def directory_receipt(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def wilson(wins: int, losses: int, draws: int, z: float = 1.96) -> list[float]:
    count = wins + losses + draws
    score = (wins + 0.5 * draws) / max(count, 1)
    denominator = 1.0 + z * z / max(count, 1)
    center = (score + z * z / (2.0 * max(count, 1))) / denominator
    half = (
        z
        * math.sqrt(
            score * (1.0 - score) / max(count, 1)
            + z * z / (4.0 * max(count, 1) ** 2)
        )
        / denominator
    )
    return [max(0.0, center - half), min(1.0, center + half)]


def counts(rows: list[dict[str, str]]) -> dict[str, Any]:
    wins = sum(row["candidate_result"] == "win" for row in rows)
    losses = sum(row["candidate_result"] == "loss" for row in rows)
    draws = sum(row["candidate_result"] == "draw" for row in rows)
    errors = sum(bool(row["error"]) for row in rows)
    return {
        "games": len(rows),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "errors": errors,
        "scoreRate": (wins + 0.5 * draws) / max(1, wins + losses + draws),
        "wilson95": wilson(wins, losses, draws),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arena-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--games-per-opponent", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidate_sha = directory_receipt(args.candidate_root.resolve())
    opponents = [row for row in manifest["opponents"] if row["promotionPool"]]
    cells: list[dict[str, Any]] = []
    all_rows: list[dict[str, str]] = []
    for opponent in opponents:
        cell_root = args.arena_root / opponent["label"]
        summary = json.loads((cell_root / "summary.json").read_text(encoding="utf-8"))
        with (cell_root / "games.csv").open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != args.games_per_opponent:
            raise RuntimeError(
                f"{opponent['label']} has {len(rows)} games, expected {args.games_per_opponent}"
            )
        if summary["candidatePackageSha256"] != candidate_sha:
            raise RuntimeError(f"candidate receipt mismatch for {opponent['label']}")
        if summary["baselinePackageSha256"] != opponent["directorySha256"]:
            raise RuntimeError(f"opponent receipt mismatch for {opponent['label']}")
        if summary["errors"] or summary["candidateAgentStats"].get("fallbackCalls", 0):
            raise RuntimeError(f"errors or fallbacks in {opponent['label']}")
        seat_zero = [row for row in rows if row["candidate_seat"] == "0"]
        seat_one = [row for row in rows if row["candidate_seat"] == "1"]
        if len(seat_zero) != len(seat_one):
            raise RuntimeError(f"seat imbalance for {opponent['label']}")
        total = counts(rows)
        cell = {
            "opponent": opponent["label"],
            "role": opponent["role"],
            "opponentDeckSha256": opponent["deckSha256"],
            "opponentPackageSha256": opponent["directorySha256"],
            "sameDeck": summary["sameDeck"],
            "total": total,
            "seat0": counts(seat_zero),
            "seat1": counts(seat_one),
            "candidateAgentStats": summary["candidateAgentStats"],
            "distinctShardFingerprints": summary["distinctShardFingerprints"],
        }
        cells.append(cell)
        all_rows.extend(rows)
    macro = sum(cell["total"]["scoreRate"] for cell in cells) / len(cells)
    result = {
        "schemaVersion": 1,
        "manifestSourcesSha256": manifest["sourcesSha256"],
        "candidateRoot": str(args.candidate_root.resolve()),
        "candidatePackageSha256": candidate_sha,
        "opponents": len(cells),
        "gamesPerOpponent": args.games_per_opponent,
        "totalGames": len(all_rows),
        "macroScoreRate": macro,
        "pooled": counts(all_rows),
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
