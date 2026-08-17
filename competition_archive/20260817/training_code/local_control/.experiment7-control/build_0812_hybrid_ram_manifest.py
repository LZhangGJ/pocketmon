#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ram-0811", type=Path, required=True)
    parser.add_argument("--tensordict-0812", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old = json.loads(args.ram_0811.resolve().read_text(encoding="utf-8"))
    new = json.loads(args.tensordict_0812.resolve().read_text(encoding="utf-8"))
    old_rows = {row["name"]: row for row in old["datasets"]}
    new_rows = {row["name"]: row for row in new["datasets"]}
    days = [f"2026-08-{day:02d}" for day in range(3, 13)]
    rows = [dict(old_rows[day]) if day != "2026-08-12" else dict(new_rows[day]) for day in days]
    for row in rows:
        for key in ("features", "tokenCache", "sequenceCache", "identityCache"):
            if not Path(row[key]).exists():
                raise FileNotFoundError(f"{row['name']} {key}: {row[key]}")
    payload = dict(new)
    payload["referenceRoot"] = str(args.reference_root.resolve())
    payload["datasets"] = rows
    payload["tensorStorage"] = {
        **new.get("tensorStorage", {}),
        "calendarDays": days,
        "runtimeLayout": "nine RAM-resident NPZ shards plus new 2026-08-12 TensorDict memmap shard",
        "ramResidentDays": days[:-1],
        "newTensorDictDay": days[-1],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "days": days, "decisions": sum(int(row["summary"]["decisions"]) for row in rows)}))


if __name__ == "__main__":
    main()
