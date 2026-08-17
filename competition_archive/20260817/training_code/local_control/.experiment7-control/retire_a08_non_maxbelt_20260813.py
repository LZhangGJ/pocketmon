#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-deck-branches-20260812")
ACTIVE = "a08_maxbelt"
RETIRED = ("a08_lilligant", "a08_lilligant_maxbelt")


def write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    at = datetime.now(timezone.utc).isoformat()
    records = []
    for branch in RETIRED:
        path = ROOT / branch
        generations = sorted(path.glob("generation-*"))
        record = {
            "branch": branch,
            "status": "retired_preserved_read_only",
            "retiredAt": at,
            "generationCount": len(generations),
            "latestGeneration": generations[-1].name if generations else None,
            "reason": "user retained only Maximum Belt A08 variant",
            "dataDeleted": False,
        }
        write(path / "RETIRED.json", record)
        records.append(record)
    payload = {
        "schemaVersion": 1,
        "updatedAt": at,
        "activeBranches": [ACTIVE],
        "retiredBranches": list(RETIRED),
        "policy": "do not launch rollout, learner, challenger, or evaluation for retired branches",
        "records": records,
    }
    write(ROOT / "control/ACTIVE_BRANCHES.json", payload)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
