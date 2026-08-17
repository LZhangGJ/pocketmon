#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--summarizer", required=True)
    parser.add_argument(
        "--section", choices=("all", "training", "full", "submission4"), default="all"
    )
    args = parser.parse_args()

    root = Path(args.league_root)
    full = json.loads((root / "monitoring/full-matrix/latest.json").read_text())
    submission = json.loads((root / "monitoring/submission4/latest.json").read_text())
    raw = subprocess.check_output(
        [args.python, "-s", args.summarizer, "--league-root", str(root)],
        text=True,
    )
    training = json.loads(raw)

    compact_training = {}
    for chain, row in training["chains"].items():
        compact_training[chain] = {
            key: row.get(key)
            for key in (
                "generation",
                "completedShards",
                "episodes",
                "decisions",
                "externalWins",
                "externalLosses",
                "selfPlayEpisodes",
                "livePpoOpponentEpisodes",
                "publishedUpdates",
                "failedUpdates",
                "latestInitialPolicyShift",
                "latestEpoch",
            )
        }

    compact_full = {}
    for chain, row in full["chains"].items():
        compact_full[chain] = {
            key: row.get(key)
            for key in (
                "generation",
                "frozenAggregate",
                "universalBcFrozenAggregate",
                "ppoMinusBc",
                "deltaVsPrevious",
                "progress",
                "seatMetrics",
                "seatGap",
                "directVsUniversalBc",
                "ppoHeadToHead",
                "agents",
            )
        }

    result = {
        "fullMatrix": {
            "status": full.get("status"),
            "updatedAt": full.get("updatedAt"),
            "roundId": full.get("roundId"),
            "games": full.get("games"),
            "engineSeedControlled": full.get("engineSeedControlled"),
            "frozenAgentCount": full.get("frozenAgentCount"),
            "chains": compact_full,
        },
        "submission4": submission,
        "training": {
            "leagueUpdatedAt": training.get("leagueUpdatedAt"),
            "chains": compact_training,
            "workerPidFiles": training.get("workerPidFiles", []),
            "legacyStopReceipts": training.get("legacyStopReceipts", []),
        },
    }
    selected = result if args.section == "all" else result[args.section if args.section != "full" else "fullMatrix"]
    print(json.dumps(selected, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
