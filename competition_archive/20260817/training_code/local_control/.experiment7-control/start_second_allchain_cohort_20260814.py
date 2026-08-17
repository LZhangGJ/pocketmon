from __future__ import annotations

import concurrent.futures
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")
HOSTS = ["doraemon02", "doraemon03", "doraemon04", "doraemon09", "doraemon10", "doraemon11", "doraemon12", "doraemon13", "doraemon14", "doraemon15", "doraemon16", "doraemon17", "doraemon19", "doraemon20"]


def start(host: str) -> dict:
    worker = f"{host}-v7b"
    command = f"/homes/lzhang/start_allchain_rollout_worker_v7_20260814.sh {worker}"
    if host == "doraemon02":
        argv = ["bash", "--noprofile", "--norc", "-c", command]
    else:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, "bash", "--noprofile", "--norc", "-c", shlex.quote(command)]
    try:
        done = subprocess.run(argv, text=True, capture_output=True, timeout=40)
        return {"host": host, "worker": worker, "returnCode": done.returncode, "output": (done.stdout + done.stderr).strip()}
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"host": host, "worker": worker, "status": "unreachable", "error": repr(exc)}


def main() -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(HOSTS)) as pool:
        rows = list(pool.map(start, HOSTS))
    receipt = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(), "cohort": "v7b", "allSevenChains": True, "rows": rows}
    path = ROOT / "control/seven-ppo-distributed-20260814/second-cohort-receipt.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
