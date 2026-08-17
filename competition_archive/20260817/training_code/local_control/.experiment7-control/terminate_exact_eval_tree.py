from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path


if len(sys.argv) != 2 or not sys.argv[1].startswith("/suedata1/Free/lzhang/pocketmon-runs/"):
    raise SystemExit("usage: terminate_exact_eval_tree.py EXACT_SCHEDULE_PATH")

marker = sys.argv[1]
commands: dict[int, str] = {}
parents: dict[int, int] = {}
for proc in Path("/proc").glob("[0-9]*"):
    try:
        pid = int(proc.name)
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        fields = (proc / "stat").read_text().split()
        commands[pid] = command
        parents[pid] = int(fields[3])
    except (OSError, PermissionError, ValueError, IndexError):
        continue

roots = {pid for pid, command in commands.items() if marker in command and pid != os.getpid()}
targets = set(roots)
changed = True
while changed:
    changed = False
    for pid, parent in parents.items():
        if parent in targets and pid not in targets:
            targets.add(pid)
            changed = True

def depth(pid: int) -> int:
    value = 0
    seen = set()
    while pid in parents and pid not in seen:
        seen.add(pid)
        pid = parents[pid]
        value += 1
    return value

ordered = sorted(targets, key=depth, reverse=True)
print(f"MARKER {marker}")
print(f"TARGET_COUNT {len(ordered)}")
for pid in ordered:
    print(f"TERM {pid} {commands.get(pid, '')[:180]}")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

time.sleep(2)
survivors = []
for pid in ordered:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        continue
    survivors.append(pid)
print("SURVIVORS", " ".join(map(str, survivors)) if survivors else "none")
