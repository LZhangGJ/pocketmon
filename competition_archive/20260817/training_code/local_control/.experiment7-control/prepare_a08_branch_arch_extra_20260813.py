#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


parser = argparse.ArgumentParser()
parser.add_argument("--branch-eval-root", type=Path, required=True)
parser.add_argument("--live-pool", type=Path, required=True)
parser.add_argument("--extra-name", default="arch-extra")
args = parser.parse_args()

pool = json.loads(args.live_pool.read_text(encoding="utf-8"))
arch = next(row for row in pool["agents"] if row["name"] == "public_archaludon_meta")
receipt = {"schemaVersion": 1, "candidates": []}
extra_root = args.branch_eval_root / args.extra_name
for candidate in sorted(args.branch_eval_root.glob("a08_*_g*")):
    league_path = candidate / "state/league.json"
    latest = candidate / "monitoring/full-matrix/latest.json"
    if not league_path.is_file() or not latest.is_file():
        continue
    extra = extra_root / candidate.name
    extra_pool = extra / "state/arch-only-pool.json"
    write(extra_pool, {**pool, "agents": [arch]})
    league = json.loads(league_path.read_text(encoding="utf-8"))
    league["poolPath"] = str(extra_pool)
    if isinstance(league.get("basePool"), dict):
        league["basePool"]["path"] = str(extra_pool)
    write(extra / "state/league.json", league)
    receipt["candidates"].append({"name": candidate.name, "root": str(extra)})
write(extra_root / "candidates.json", receipt)
print(json.dumps(receipt, ensure_ascii=False))
