from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a new pool revision with one hash-identical Agent source")
    parser.add_argument("--base-pool", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--replacement-dir", type=Path, required=True)
    parser.add_argument("--integration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    integration_root = args.integration_root.resolve()
    sys.path.insert(0, str(integration_root))
    from common import directory_sha256  # noqa: PLC0415

    base_path = args.base_pool.resolve()
    output = args.output.resolve()
    receipt = args.receipt.resolve()
    replacement = args.replacement_dir.resolve()
    if output.exists() or receipt.exists():
        raise FileExistsError(f"refusing to overwrite pool revision: {output} / {receipt}")
    base_hash = sha256_file(base_path)
    if base_hash != args.expected_base_sha256.lower():
        raise ValueError(f"base pool hash mismatch: {base_hash}")
    payload = json.loads(base_path.read_text(encoding="utf-8"))
    agents = payload.get("agents", [])
    matches = [row for row in agents if row.get("name") == args.agent_name]
    if len(matches) != 1:
        raise ValueError(f"Agent is not unique in base pool: {args.agent_name}")
    target = matches[0]
    expected = str(target["directory_sha256"]).lower()
    actual = directory_sha256(replacement)
    if actual != expected:
        raise ValueError(f"replacement directory hash mismatch: expected={expected} actual={actual}")
    previous_dir = Path(str(target["agent_dir"]))
    previous_actual = directory_sha256(previous_dir)
    target["agent_dir"] = str(replacement)

    integrity = []
    for row in agents:
        agent_dir = Path(str(row["agent_dir"]))
        row_expected = str(row["directory_sha256"]).lower()
        row_actual = directory_sha256(agent_dir)
        integrity.append(
            {
                "name": row["name"],
                "agentDir": str(agent_dir),
                "expectedDirectorySha256": row_expected,
                "actualDirectorySha256": row_actual,
                "matches": row_expected == row_actual,
            }
        )
    mismatches = [row for row in integrity if not row["matches"]]
    if mismatches:
        raise ValueError(f"pool revision still has hash mismatches: {mismatches}")

    created_at = datetime.now(timezone.utc).isoformat()
    payload["schema"] = "experiment7_frozen_opponent_pool_v3r1"
    payload["schemaVersion"] = 3
    payload["revision"] = 1
    payload["createdAt"] = created_at
    payload["sources"] = {
        **payload.get("sources", {}),
        "integrityRepair": {
            "basePool": {"path": str(base_path), "sha256": base_hash},
            "agent": args.agent_name,
            "previousAgentDir": str(previous_dir),
            "previousActualDirectorySha256": previous_actual,
            "replacementAgentDir": str(replacement),
            "replacementDirectorySha256": actual,
            "contentIdentityPreserved": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    output_hash = sha256_file(output)
    audit = {
        "schemaVersion": 1,
        "createdAt": created_at,
        "basePool": {"path": str(base_path), "sha256": base_hash},
        "newPool": {"path": str(output), "sha256": output_hash, "agents": len(agents)},
        "repointedAgent": {
            "name": args.agent_name,
            "previousDir": str(previous_dir),
            "previousActualDirectorySha256": previous_actual,
            "replacementDir": str(replacement),
            "directorySha256": actual,
        },
        "integrity": integrity,
        "allDirectoryHashesMatch": True,
        "basePoolModified": False,
    }
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
