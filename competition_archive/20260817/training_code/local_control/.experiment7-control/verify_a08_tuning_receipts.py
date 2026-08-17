import json
from pathlib import Path

root = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")
batch_path = root / "learners/a08_dipplin_seaking/generation-000350/batch.json"
summary_path = root / "buffer/ready/a08_dipplin_seaking/doraemon10-00001063-1786582452739453458.jsonl.gz.summary.json"
batch = json.loads(batch_path.read_text())
summary = json.loads(summary_path.read_text())
print(json.dumps({
    "batch": {
        "path": str(batch_path),
        "createdAt": batch.get("createdAt"),
        "decisions": batch.get("decisions"),
        "rolloutCount": len(batch.get("rollouts", [])),
        "targetedRollouts": sum("a08-targeted" in path for path in batch.get("rollouts", [])),
        "trainingControl": batch.get("trainingControl"),
    },
    "rollout": {
        "path": str(summary_path),
        "createdAt": summary.get("createdAt"),
        "episodes": summary.get("episodes"),
        "decisions": summary.get("decisions"),
        "samplingControl": summary.get("samplingControl"),
    },
}, ensure_ascii=False))
