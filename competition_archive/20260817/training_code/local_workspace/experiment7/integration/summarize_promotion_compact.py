from __future__ import annotations

import json
import sys


CHAINS = (
    "a02_grim_large_g9_pokegear",
    "a08_maxbelt_large_g9",
    "dragapult_munkidori_large_g9",
    "lucario_gold_exact",
    "alakazam_large_g9",
    "kangaskhan_crustle_large_g9",
    "festival_grass_large_g9",
)


def main() -> int:
    payload = json.load(sys.stdin)
    candidates = payload.get("candidates", {})
    admissions = payload.get("admissions", {})
    result = {"updatedAt": payload.get("updatedAt"), "chains": {}}
    for chain in CHAINS:
        candidate = candidates.get(chain) or {}
        admission = admissions.get(chain) or {}
        admitted_candidate = admission.get("candidate") or {}
        observations = candidate.get("observations") or []
        latest = observations[-1] if observations else {}
        frozen = latest.get("frozenAggregate") or {}
        direct = latest.get("directVsUniversalBc") or {}
        result["chains"][chain] = {
            "candidateGeneration": candidate.get("generation"),
            "candidateSnapshotId": candidate.get("snapshotId"),
            "status": candidate.get("status"),
            "consecutivePasses": len(candidate.get("passRounds") or []),
            "latestRound": latest.get("roundId"),
            "latestPassed": latest.get("passed"),
            "failureReasons": latest.get("failureReasons") or latest.get("reasons") or [],
            "frozenCompleted": frozen.get("completed"),
            "frozenScoreRate": frozen.get("scoreRate"),
            "directCompleted": direct.get("completed"),
            "directScoreRate": direct.get("scoreRate"),
            "seatGap": latest.get("seatGap"),
            "admittedGeneration": admitted_candidate.get("generation"),
            "admittedSnapshotId": admitted_candidate.get("snapshotId"),
        }
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
