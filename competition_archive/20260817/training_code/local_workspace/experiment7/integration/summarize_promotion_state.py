#!/usr/bin/env python3
import json
import sys


def main() -> None:
    data = json.load(sys.stdin)
    candidates = {}
    for chain, item in data.get("candidates", {}).items():
        candidates[chain] = {
            key: item.get(key)
            for key in (
                "generation",
                "snapshotId",
                "status",
                "passRounds",
                "firstAdmission",
                "supersededAt",
                "supersededBy",
            )
        }
        observations = item.get("observations", [])
        candidates[chain]["latestObservation"] = observations[-1] if observations else None
    result = {
        "updatedAt": data.get("updatedAt"),
        "candidates": candidates,
        "admissions": data.get("admissions", {}),
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
