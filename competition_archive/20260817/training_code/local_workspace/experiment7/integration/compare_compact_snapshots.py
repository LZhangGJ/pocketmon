from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("previous", type=Path)
    parser.add_argument("current", type=Path)
    args = parser.parse_args()
    previous = json.loads(args.previous.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))
    result = {}
    for name, row in current.get("chains", {}).items():
        old = previous.get("chains", {}).get(name, {})
        result[name] = {
            "generation": row.get("generation"),
            "generationDelta": row.get("generation", 0) - old.get("generation", row.get("generation", 0)),
            "shardsDelta": row.get("completedShards", 0) - old.get("completedShards", row.get("completedShards", 0)),
            "episodesDelta": row.get("episodes", 0) - old.get("episodes", row.get("episodes", 0)),
            "decisionsDelta": row.get("decisions", 0) - old.get("decisions", row.get("decisions", 0)),
            "publishedDelta": row.get("publishedUpdates", 0) - old.get("publishedUpdates", row.get("publishedUpdates", 0)),
            "failedDelta": row.get("failedUpdates", 0) - old.get("failedUpdates", row.get("failedUpdates", 0)),
            "currentGenerationExternal": row.get("currentGenerationExternal"),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
