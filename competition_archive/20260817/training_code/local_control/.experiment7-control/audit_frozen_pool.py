from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only audit of frozen Agent directory hashes")
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--integration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    integration_root = args.integration_root.resolve()
    sys.path.insert(0, str(integration_root))
    from common import directory_sha256  # noqa: PLC0415

    pool_path = args.pool.resolve()
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    rows = payload.get("agents")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"pool has no agents: {pool_path}")
    audit_rows = []
    for row in rows:
        name = str(row.get("name", ""))
        agent_dir = Path(str(first(row, "agent_dir", "agentDir", "path")))
        expected = str(first(row, "directory_sha256", "directorySha256") or "").lower()
        error = ""
        actual = ""
        try:
            actual = directory_sha256(agent_dir)
        except Exception as exc:  # preserve the full pool audit when one source is unreadable
            error = f"{type(exc).__name__}: {exc}"
        audit_rows.append(
            {
                "name": name,
                "agentDir": str(agent_dir),
                "expectedDirectorySha256": expected,
                "actualDirectorySha256": actual,
                "matches": bool(expected) and expected == actual,
                "error": error,
            }
        )
    result = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "pool": str(pool_path),
        "agents": audit_rows,
        "matched": sum(row["matches"] for row in audit_rows),
        "mismatched": sum(not row["matches"] for row in audit_rows),
        "readOnlyAudit": True,
    }
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False))
    if result["mismatched"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
