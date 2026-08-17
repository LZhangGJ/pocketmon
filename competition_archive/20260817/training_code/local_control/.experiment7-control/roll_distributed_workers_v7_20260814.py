from __future__ import annotations

import concurrent.futures
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")
HOSTS = [
    "doraemon02", "doraemon03", "doraemon04", "doraemon08", "doraemon09",
    "doraemon10", "doraemon11", "doraemon12", "doraemon13", "doraemon14",
    "doraemon15", "doraemon16", "doraemon17", "doraemon19", "doraemon20",
]
STARTER = "/homes/lzhang/start_allchain_rollout_worker_v7_20260814.sh"


def remote(host: str, command: str, timeout: int = 25) -> subprocess.CompletedProcess[str]:
    if host == "doraemon02":
        argv = ["bash", "--noprofile", "--norc", "-c", command]
    else:
        argv = [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host,
            "bash", "--noprofile", "--norc", "-c", shlex.quote(command),
        ]
    return subprocess.run(argv, text=True, capture_output=True, timeout=timeout)


def roll(host: str) -> dict:
    pidfile = ROOT / "workers" / f"{host}.pid"
    old_pid = pidfile.read_text().strip() if pidfile.is_file() else ""
    result = {"host": host, "workerId": host, "oldPid": old_pid, "status": "unknown"}
    try:
        probe = remote(host, "hostname; awk '/^cpu /{print}' /proc/stat; cat /proc/pressure/io | head -1")
    except (subprocess.TimeoutExpired, OSError) as exc:
        result["status"] = "unreachable"
        result["error"] = repr(exc)
        return result
    result["probe"] = (probe.stdout + probe.stderr).strip()
    if probe.returncode != 0:
        result["status"] = "unreachable"
        return result
    if old_pid.isdigit():
        try:
            check = remote(host, f"tr '\\0' ' ' </proc/{old_pid}/cmdline 2>/dev/null || true")
        except (subprocess.TimeoutExpired, OSError) as exc:
            result["status"] = "unreachable"
            result["error"] = repr(exc)
            return result
        cmdline = check.stdout.strip()
        result["oldCommand"] = cmdline
        if "run_async_ppo_rollout_worker.py" in cmdline and f"--worker-id {host}" in cmdline:
            remote(host, f"kill -TERM {old_pid}")
            for _ in range(20):
                alive = remote(host, f"kill -0 {old_pid} 2>/dev/null").returncode == 0
                if not alive:
                    break
                time.sleep(0.25)
            result["oldStopped"] = not alive
            if alive:
                result["status"] = "deferred_old_worker_still_exiting"
                return result
        elif cmdline:
            result["status"] = "refused_pid_mismatch"
            return result
    try:
        start = remote(host, f"{STARTER} {host}", timeout=40)
    except (subprocess.TimeoutExpired, OSError) as exc:
        result["status"] = "start_timeout"
        result["error"] = repr(exc)
        return result
    result["startOutput"] = (start.stdout + start.stderr).strip()
    result["status"] = "started" if start.returncode == 0 else "start_failed"
    if pidfile.is_file():
        result["newPid"] = pidfile.read_text().strip()
    return result


def main() -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(HOSTS)) as pool:
        rows = list(pool.map(roll, HOSTS))
    receipt = {
        "schemaVersion": 1,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "launcher": STARTER,
        "allActivePpoChains": True,
        "cpuLimitPercent": 95,
        "ioPressureLimitPercent": 80,
        "rows": rows,
    }
    path = ROOT / "control/seven-ppo-distributed-20260814/worker-roll-receipt.json"
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
