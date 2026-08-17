from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely rebuild one audited public Agent candidate")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--expected-directory-sha256", required=True)
    parser.add_argument("--integration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    integration_root = args.integration_root.resolve()
    sys.path.insert(0, str(integration_root))
    from common import sha256_file  # noqa: PLC0415
    from stage_opponent_pool import (  # noqa: PLC0415
        PUBLIC_ARCHIVE_SHA256,
        PUBLIC_CANDIDATES,
        _ensure_new_staging_root,
        _require_deck_identity,
        _require_hash,
        _staged_row,
        copy_public_candidate,
        static_scan_agent,
    )

    archive = args.archive.resolve()
    output_root = args.output_root.resolve()
    _require_hash(archive, PUBLIC_ARCHIVE_SHA256, "public agent archive")
    matches = [row for row in PUBLIC_CANDIDATES if row["name"] == args.candidate]
    if len(matches) != 1:
        raise ValueError(f"candidate is not uniquely defined: {args.candidate}")
    candidate = matches[0]
    _ensure_new_staging_root(output_root)
    target = output_root / "agents" / candidate["name"]
    copy_public_candidate(archive, candidate, target)
    scan = static_scan_agent(target)
    _require_deck_identity(candidate, scan)
    expected_directory_hash = args.expected_directory_sha256.lower()
    if scan["directorySha256"] != expected_directory_hash:
        raise ValueError(
            "rebuilt directory hash does not match the historical frozen hash: "
            f"expected={expected_directory_hash} actual={scan['directorySha256']}"
        )
    staged = _staged_row(candidate, target, scan, "public_archive_rebuild")
    packages = {
        "schemaVersion": 1,
        "packages": [
            {
                "name": candidate["name"],
                "agentDir": str(target),
                "status": "staging",
                "archetype": candidate["archetype"],
                "deckCanonicalSha256": scan["deckCanonicalSha256"],
                "directorySha256": scan["directorySha256"],
            }
        ],
    }
    packages_path = output_root / "packages.json"
    with packages_path.open("x", encoding="utf-8") as handle:
        json.dump(packages, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    manifest = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "status": "static_rebuild_only_not_arena_admitted",
        "source": {"path": str(archive), "sha256": PUBLIC_ARCHIVE_SHA256},
        "stagingRoot": str(output_root),
        "agents": [staged],
        "packages": {"path": str(packages_path), "sha256": sha256_file(packages_path)},
        "externalAgentCodeExecuted": False,
        "historicalDirectorySha256Matched": True,
    }
    manifest_path = output_root / "staging_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
