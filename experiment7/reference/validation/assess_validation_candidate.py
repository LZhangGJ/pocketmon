from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--min-macro-delta", type=float, default=0.03)
    parser.add_argument("--max-cell-harm", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    control = json.loads(args.control.read_text(encoding="utf-8"))
    candidate_cells = {row["opponent"]: row for row in candidate["cells"]}
    control_cells = {row["opponent"]: row for row in control["cells"]}
    if set(candidate_cells) != set(control_cells):
        raise RuntimeError("candidate/control opponent sets differ")
    cells = []
    for opponent in control_cells:
        candidate_rate = float(candidate_cells[opponent]["total"]["scoreRate"])
        control_rate = float(control_cells[opponent]["total"]["scoreRate"])
        delta = candidate_rate - control_rate
        cells.append(
            {
                "opponent": opponent,
                "candidateScoreRate": candidate_rate,
                "controlScoreRate": control_rate,
                "delta": delta,
                "passesHarmGate": delta >= -args.max_cell_harm,
            }
        )
    macro_delta = float(candidate["macroScoreRate"]) - float(control["macroScoreRate"])
    no_errors = int(candidate["pooled"]["errors"]) == 0 and all(
        int(row["candidateAgentStats"].get("fallbackCalls", 0)) == 0
        for row in candidate["cells"]
    )
    passes = (
        no_errors
        and macro_delta >= args.min_macro_delta
        and all(row["passesHarmGate"] for row in cells)
    )
    payload = {
        "schemaVersion": 1,
        "candidateMatrix": {
            "path": str(args.candidate.resolve()),
            "sha256": sha256(args.candidate),
            "gamesPerOpponent": candidate["gamesPerOpponent"],
            "candidatePackageSha256": candidate["candidatePackageSha256"],
        },
        "controlMatrix": {
            "path": str(args.control.resolve()),
            "sha256": sha256(args.control),
            "gamesPerOpponent": control["gamesPerOpponent"],
            "candidatePackageSha256": control["candidatePackageSha256"],
        },
        "frozenRule": {
            "minMacroDelta": args.min_macro_delta,
            "maxCellHarm": args.max_cell_harm,
            "zeroErrorsAndFallbacks": True,
        },
        "candidateMacroScoreRate": candidate["macroScoreRate"],
        "controlMacroScoreRate": control["macroScoreRate"],
        "macroDelta": macro_delta,
        "zeroErrorsAndFallbacks": no_errors,
        "cells": cells,
        "passesPromotionGate": passes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
