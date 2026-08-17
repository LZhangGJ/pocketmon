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
        raise TypeError(f"expected JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an additive PPO training pool without mutating the frozen base pool."
    )
    parser.add_argument("--base-pool", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_path = args.base_pool.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    payload = load_json(base_path)
    agents = payload.get("agents")
    if not isinstance(agents, list):
        raise TypeError("base pool must contain an agents list")

    seen = {str(row["name"]) for row in agents}
    additions: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for manifest_arg in args.package_manifest:
        manifest_path = manifest_arg.resolve()
        manifest = load_json(manifest_path)
        packages = manifest.get("packages")
        if not isinstance(packages, list) or not packages:
            raise ValueError(f"package manifest has no packages: {manifest_path}")
        sources.append({"path": str(manifest_path), "sha256": sha256_file(manifest_path)})
        for package in packages:
            name = str(package["name"])
            if name in seen:
                raise ValueError(f"duplicate agent name: {name}")
            agent_dir = Path(package["agentDir"]).resolve()
            for required in (agent_dir / "main.py", agent_dir / "deck.csv"):
                if not required.is_file():
                    raise FileNotFoundError(required)
            archetype_id = str(package["archetypeId"])
            additions.append(
                {
                    "name": name,
                    "agent_dir": str(agent_dir),
                    "status": "accepted",
                    "pool_status": "admitted_verified_ppo_champion",
                    "archetype": archetype_id,
                    "archetype_label": str(package["archetypeLabel"]),
                    "deck_canonical_sha256": str(package["deckSha256"]),
                    "directory_sha256": str(package["directorySha256"]),
                    "skill_tier": "hard",
                    "policy_weight_within_archetype": 1.0,
                    "source_package_manifest": str(manifest_path),
                }
            )
            seen.add(name)

    payload["agents"] = [*agents, *additions]
    payload["trainingPoolAugmentation"] = {
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "basePool": {"path": str(base_path), "sha256": sha256_file(base_path)},
        "packageManifests": sources,
        "addedAgents": len(additions),
        "policy": "add all six deck packages from hard g10 and diversity g20; mark as hard",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload["trainingPoolAugmentation"], ensure_ascii=False))
    print(f"agents={len(payload['agents'])}")
    print(f"output={output_path}")
    print(f"sha256={sha256_file(output_path)}")


if __name__ == "__main__":
    main()
