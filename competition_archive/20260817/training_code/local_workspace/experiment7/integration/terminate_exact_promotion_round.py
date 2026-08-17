from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path


ALLOWED_ROOT = Path(
    "/dataT0/Free/lzhang/pocketmon-runs/"
    "experiment7-async-ppo-league-20260811/monitoring/ppo-frozen-promotion/rounds"
)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: terminate_exact_promotion_round.py EXACT_ROUND_PATH")

    marker_path = Path(sys.argv[1])
    if marker_path.parent != ALLOWED_ROOT or not marker_path.name.startswith("."):
        raise SystemExit(f"refusing marker outside an in-progress promotion round: {marker_path}")
    marker = str(marker_path)

    commands: dict[int, str] = {}
    parents: dict[int, int] = {}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc.name)
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace"
            )
            fields = (proc / "stat").read_text().split()
            commands[pid] = command
            parents[pid] = int(fields[3])
        except (OSError, PermissionError, ValueError, IndexError):
            continue

    roots = {pid for pid, command in commands.items() if marker in command and pid != os.getpid()}
    targets = set(roots)
    while True:
        before = len(targets)
        targets.update(pid for pid, parent in parents.items() if parent in targets)
        if len(targets) == before:
            break

    def depth(pid: int) -> int:
        value = 0
        seen: set[int] = set()
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
    survivors: list[int] = []
    for pid in ordered:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        survivors.append(pid)
    print("SURVIVORS", " ".join(map(str, survivors)) if survivors else "none")
    return 1 if survivors else 0


if __name__ == "__main__":
    raise SystemExit(main())
