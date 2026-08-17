#!/usr/bin/env python3
import datetime as dt
import json
import os
import pathlib
import signal
import time

ROOT = pathlib.Path(
    "/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811"
)
ROUND = ROOT / "monitoring/urgent-targeted-compare/rounds/20260814T0836JST-g271-g271-g295-attempt2"


def cmd(pid):
    try:
        return pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, ProcessLookupError):
        return ""


def kids(pid):
    try:
        path = pathlib.Path(f"/proc/{pid}/task/{pid}/children")
        return [int(value) for value in path.read_text().split()]
    except (FileNotFoundError, ProcessLookupError):
        return []


controllers = []
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    pid = int(entry.name)
    if cmd(pid).strip() == "/usr/bin/python3 /homes/lzhang/launch_urgent_targeted_d16.py":
        controllers.append(pid)

targets = []
stack = list(controllers)
while stack:
    pid = stack.pop()
    if pid in targets:
        continue
    targets.append(pid)
    stack.extend(kids(pid))

for pid in reversed(targets):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

time.sleep(5)
for pid in reversed(targets):
    if cmd(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

receipt = json.loads((ROUND / "receipt.json").read_text())
receipt.update(
    {
        "status": "aborted_before_results",
        "stoppedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": "shared package reads entered rpc_wait; zero result rows; relaunch requires node-local package cache",
        "stoppedProcessCount": len(targets),
    }
)
for path in (ROUND / "receipt.json", ROOT / "monitoring/urgent-targeted-compare/active.json"):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(tmp, path)
