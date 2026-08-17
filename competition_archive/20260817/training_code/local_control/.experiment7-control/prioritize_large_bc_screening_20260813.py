from __future__ import annotations

import importlib.util
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


CONTROLLER = Path("/homes/lzhang/run_bc_replacement_screening_20260813.py")
LOCAL = Path("/tmp/experiment7-bc-replacement-screen-20260813")
SHARED = Path(
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/"
    "0812-d14-ram-npz-fast-20260813/replacement-screening"
)
GUARDED_RUNNER = Path("/homes/lzhang/run_load_guarded_arena_shard.sh")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parity_passed(payload: dict) -> bool:
    return all(
        int(payload.get(key, -1)) == 0
        for key in ("actionMismatches", "stableRankingMismatches", "illegalPredictionCount")
    )


def main() -> None:
    local_parity = LOCAL / "large_256x6/parity.json"
    while not local_parity.is_file():
        time.sleep(5)
    parity = json.loads(local_parity.read_text(encoding="utf-8"))
    if not parity_passed(parity):
        atomic_json(
            SHARED / "LARGE_PRIORITY_BLOCKED.json",
            {"status": "parity_failed", "parity": parity},
        )
        raise SystemExit("large parity still fails")

    target = SHARED / "large_256x6"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_parity, target / "parity.json")
    atomic_json(
        SHARED / "PARITY_RESOLVED.json",
        {
            "status": "resolved",
            "resolvedAt": datetime.now(timezone.utc).isoformat(),
            "reason": "PyTorch and portable now use the same 5e-4 near-tie rule",
            "largeParity": str((target / "parity.json").resolve()),
            "previousFailureRetained": str((SHARED / "PARITY_FAILED.json").resolve()),
        },
    )

    spec = importlib.util.spec_from_file_location("bc_replacement", CONTROLLER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {CONTROLLER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN_SHARD = GUARDED_RUNNER
    portable = target / "universal_bc.npz"
    module.run_stage("large_256x6-smoke", portable, 2, 16)
    module.run_stage("large_256x6-frozen40", portable, 40, 45)

    standard_parity = SHARED / "standard_1m/parity.json"
    if not standard_parity.is_file():
        raise FileNotFoundError(standard_parity)
    atomic_json(
        SHARED / "READY.json",
        {
            "schemaVersion": 1,
            "status": "parity_passed",
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "priorityOrder": ["large_256x6", "standard_1m", "current_bc"],
            "profiles": {
                "standard_1m": {"parity": str(standard_parity.resolve())},
                "large_256x6": {"parity": str((target / "parity.json").resolve())},
            },
        },
    )


if __name__ == "__main__":
    main()
