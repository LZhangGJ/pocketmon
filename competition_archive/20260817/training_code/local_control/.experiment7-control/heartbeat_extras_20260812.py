#!/usr/bin/env python3
import json
from pathlib import Path

def load(path):
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        return {"error": str(exc)}

def tail(path, n=10):
    try:
        return path.read_text(errors="replace").replace("\x00", "").splitlines()[-n:]
    except Exception:
        return []

out = {"branches": {}, "capacity": {}}
branch_root = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-a08-deck-branches-20260812")
for name in ("a08_maxbelt", "a08_lilligant", "a08_lilligant_maxbelt"):
    metrics = sorted((branch_root / name).glob("generation-*/metrics.json"))
    out["branches"][name] = {
        "generationCount": len(metrics),
        "latest": load(metrics[-1]) if metrics else None,
        "logTail": tail(branch_root / "logs" / f"{name}.log", 8),
    }

cap = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/capacity-comparison")
out["capacity"]["controller"] = tail(cap / "controller.log", 12)
out["capacity"]["exit"] = load(cap / "control/exit-status.json") if (cap / "control/exit-status.json").exists() else None
for name in ("standard_1m", "large_256x6"):
    root = cap / name
    reports = [p for p in root.rglob("*.json") if p.name in ("training_report.json", "summary.json", "metrics.json")]
    checkpoints = sorted(root.rglob("*.pt"))
    out["capacity"][name] = {
        "reports": {str(p.relative_to(root)): load(p) for p in reports[-5:]},
        "checkpointCount": len(checkpoints),
        "latestCheckpoint": str(checkpoints[-1].relative_to(root)) if checkpoints else None,
        "logTail": tail(cap / "logs" / f"{name}.log", 16),
    }
print(json.dumps(out, ensure_ascii=False))
