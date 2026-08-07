from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def audit(path: Path) -> dict[str, object]:
    rows = 0
    episodes: set[str] = set()
    policy_rows = 0
    value_rows = 0
    positive_rows = 0
    negative_rows = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            episodes.add(str(row["episode_id"]))
            policy_rows += int(float(row.get("policy_weight", 0.0)) > 0)
            value_rows += int(float(row.get("value_weight", 0.0)) > 0)
            positive_rows += int(float(row.get("outcome", 0.0)) > 0)
            negative_rows += int(float(row.get("outcome", 0.0)) < 0)
    return {
        "path": str(path),
        "rows": rows,
        "episodes": len(episodes),
        "policy_rows": policy_rows,
        "value_rows": value_rows,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a compressed schema-v2 RL trajectory shard")
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([audit(path) for path in args.inputs], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
