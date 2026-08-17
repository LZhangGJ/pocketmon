from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_portable_bundle(source: Path, generation: int) -> dict[str, Any]:
    if generation == 0:
        package_root = source.parent.parent
        success = next(
            (
                marker
                for marker in (package_root / "PACKAGE_SUCCESS", package_root / "SUCCESS")
                if marker.exists()
            ),
            None,
        )
        if success is None:
            raise FileNotFoundError(f"g0 package success marker is missing: {package_root}")
        return {"kind": "bc_g0", "successMarker": str(success)}

    bundle_root = source.parent.parent
    success = bundle_root / "SUCCESS"
    parity_path = bundle_root / "portable_parity.json"
    if not success.exists():
        raise FileNotFoundError(f"PPO package success marker is missing: {success}")
    if not parity_path.is_file():
        raise FileNotFoundError(f"PPO parity report is missing: {parity_path}")
    parity = load_json(parity_path)
    required_zero = (
        "actionMismatches",
        "stableRankingMismatches",
        "illegalPredictionCount",
    )
    bad = {key: parity.get(key) for key in required_zero if parity.get(key) != 0}
    if bad:
        raise ValueError(f"PPO portable parity gate failed for {bundle_root}: {bad}")
    decisions = int(parity.get("decisions", 0))
    if decisions < 1200:
        raise ValueError(f"PPO portable parity coverage is too small: {decisions} < 1200")
    return {
        "kind": "ppo",
        "successMarker": str(success),
        "parity": {
            "path": str(parity_path),
            "sha256": sha256_file(parity_path),
            "decisions": decisions,
            **{key: parity[key] for key in required_zero},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a frozen role-specific PPO Arena package manifest"
    )
    parser.add_argument(
        "--entry",
        nargs=4,
        action="append",
        metavar=("ROLE", "GENERATION", "PACKAGES_JSON", "PACKAGE_NAME"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--learners-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    learners_output = args.learners_output.resolve()
    receipt = args.receipt.resolve()
    if output.exists() or learners_output.exists() or receipt.exists():
        raise FileExistsError(
            f"refusing to overwrite: {output} / {learners_output} / {receipt}"
        )

    packages: list[dict[str, Any]] = []
    audit_entries: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_role_generations: set[tuple[str, int]] = set()
    for role, generation_text, source_text, package_name in args.entry:
        generation = int(generation_text)
        if generation not in {0, 10, 20, 30, 40}:
            raise ValueError(f"only g0/g10/g20/g30/g40 are allowed, got g{generation}: {role}")
        role_generation = (role, generation)
        if role_generation in seen_role_generations:
            raise ValueError(f"duplicate role/generation: {role_generation}")
        seen_role_generations.add(role_generation)

        source = Path(source_text).resolve()
        payload = load_json(source)
        rows = payload.get("packages")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"source has no packages: {source}")
        matches = [row for row in rows if row.get("name") == package_name]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one package named {package_name!r} in {source}, got {len(matches)}"
            )
        row = dict(matches[0])
        name = str(row.get("name", ""))
        if not name or name in seen_names:
            raise ValueError(f"invalid or duplicate Arena package name: {name!r}")
        seen_names.add(name)
        if not row.get("agentDir") or len(str(row.get("directorySha256", ""))) != 64:
            raise ValueError(f"package is missing frozen path/hash: {name}")

        validation = validate_portable_bundle(source, generation)
        row["status"] = "accepted"
        row["experiment7Role"] = role
        row["experiment7Generation"] = generation
        packages.append(row)
        audit_entries.append(
            {
                "role": role,
                "generation": generation,
                "packageName": name,
                "source": {"path": str(source), "sha256": sha256_file(source)},
                "agentDir": row["agentDir"],
                "directorySha256": row["directorySha256"],
                "portableValidation": validation,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    learners_output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump({"schemaVersion": 1, "packages": packages}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    learners = []
    for row in packages:
        learner = {
            "name": row["name"],
            "agent_dir": row["agentDir"],
            "directory_sha256": row["directorySha256"],
            "status": "accepted",
            "experiment7Role": row["experiment7Role"],
            "experiment7Generation": row["experiment7Generation"],
        }
        for key in ("archetypeId", "archetypeLabel", "deckSha256"):
            if key in row:
                learner[key] = row[key]
        learners.append(learner)
    with learners_output.open("x", encoding="utf-8") as handle:
        json.dump({"schemaVersion": 1, "agents": learners}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    audit = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "learnersOutput": {
            "path": str(learners_output),
            "sha256": sha256_file(learners_output),
        },
        "entries": audit_entries,
        "entryCount": len(audit_entries),
        "sourcePackagesModified": False,
    }
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
