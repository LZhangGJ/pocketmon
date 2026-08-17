from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def atomic_write(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("chains", nargs="+")
    args = parser.parse_args()
    cutoff = parse_time(args.cutoff)
    restored: dict[str, list[str]] = {}

    for chain in args.chains:
        chain_root = args.run_root / chain
        ledger_path = chain_root / "rollout-ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        failed_rollouts: set[str] = set()
        for failure in chain_root.glob("generation-*/FAILED.json"):
            if datetime.fromtimestamp(failure.stat().st_mtime, timezone.utc) < cutoff:
                continue
            batch_path = failure.parent / "batch.json"
            if not batch_path.is_file():
                continue
            batch = json.loads(batch_path.read_text(encoding="utf-8"))
            failed_rollouts.update(str(Path(path).resolve()) for path in batch.get("rollouts", []))

        removed = []
        rejected = ledger.setdefault("rejected", {})
        for path in sorted(failed_rollouts):
            row = rejected.get(path)
            if not isinstance(row, dict):
                continue
            if row.get("reason") != "trainer_failed_or_policy_shift_gate":
                continue
            if parse_time(str(row["recordedAt"])) < cutoff:
                continue
            del rejected[path]
            removed.append(path)
        if removed:
            atomic_write(ledger_path, ledger)
        restored[chain] = removed

    receipt = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "cutoff": cutoff.isoformat(),
        "restored": restored,
        "restoredCount": sum(len(paths) for paths in restored.values()),
    }
    atomic_write(args.receipt, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
