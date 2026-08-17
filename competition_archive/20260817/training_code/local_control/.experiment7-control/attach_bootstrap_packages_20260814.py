from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")
INTEGRATION = Path("/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0/experiment7/integration")
sys.path.insert(0, str(INTEGRATION))
from async_ppo_control import atomic_write_json, build_pool_payload, read_json, sha256_file, state_lock, utc_now  # noqa: E402


CHAINS = ["universal_ppo_standard_1m", "universal_ppo_large_256x6", "lucario_gold_exact"]


def main() -> None:
    league_path = ROOT / "state/league.json"
    attached = []
    with state_lock(league_path.with_suffix(league_path.suffix + ".lock")):
        league = read_json(league_path)
        for name in CHAINS:
            current = league["chains"][name]["current"]
            if current.get("packageManifest"):
                continue
            manifest = ROOT / f"learners/{name}/generation-000000-bootstrap/deployment/packages/packages.json"
            payload = read_json(manifest)
            candidates = [
                row for row in payload.get("packages", [])
                if str(row.get("archetypeId", "")).upper()
                == str(league["chains"][name]["archetypeId"]).upper()
            ]
            if len(candidates) != 1:
                raise RuntimeError(f"invalid bootstrap manifest for {name}: {manifest}")
            current["packageManifest"] = str(manifest)
            current["publishedAt"] = utc_now()
            attached.append(name)
        league["updatedAt"] = utc_now()
        pool_path = Path(league["poolPath"])
        atomic_write_json(pool_path, build_pool_payload(league))
        league["poolSha256"] = sha256_file(pool_path)
        atomic_write_json(league_path, league)
    receipt = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "status": "attached",
        "chains": attached,
        "reason": "shared checkpoint RPC stall; portable exported from exact d14 local cache",
    }
    path = ROOT / "control/seven-ppo-distributed-20260814/bootstrap-attach-receipt.json"
    atomic_write_json(path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
