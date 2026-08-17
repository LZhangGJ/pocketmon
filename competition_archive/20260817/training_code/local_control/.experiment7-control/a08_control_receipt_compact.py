import json
from pathlib import Path

root = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")
league = json.loads((root / "state/league.json").read_text())
name = "a08_dipplin_seaking"
summaries = sorted(root.glob(f"buffer/**/{name}/**/*.summary.json"), key=lambda p: p.stat().st_mtime)
batches = sorted(root.glob(f"learners/{name}/**/batch.json"), key=lambda p: p.stat().st_mtime)
summary_path = summaries[-1] if summaries else None
batch_path = batches[-1] if batches else None
summary = json.loads(summary_path.read_text()) if summary_path else {}
batch = json.loads(batch_path.read_text()) if batch_path else {}
print(json.dumps({
    "leagueControl": league["chains"][name].get("trainingControl"),
    "summaryPath": str(summary_path),
    "summary": summary,
    "batchPath": str(batch_path),
    "batch": batch,
}, ensure_ascii=False))
