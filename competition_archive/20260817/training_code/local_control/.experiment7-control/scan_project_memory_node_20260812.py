#!/usr/bin/env python3
import glob
import os

categories = {"bc": 0, "ppo_train": 0, "rollout": 0, "eval": 0, "other": 0}
counts = {key: 0 for key in categories}
for cmdline in glob.glob("/proc/[0-9]*/cmdline"):
    try:
        raw = open(cmdline, "rb").read().decode(errors="replace").replace("\0", " ")
        if not any(token in raw for token in ("pocketmon-runs/experiment7", "worktrees/experiment7", "pocketmon/experiment7")):
            continue
        pages = int(open(cmdline.replace("cmdline", "statm")).read().split()[1])
        rss = pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        continue
    if "train_universal_bc.py" in raw:
        category = "bc"
    elif "run_async_ppo_learner.py" in raw or "train_universal_ppo.py" in raw:
        category = "ppo_train"
    elif "collect_universal_ppo_rollouts.py" in raw or "run_async_ppo_rollout_worker.py" in raw:
        category = "rollout"
    elif "arena" in raw.lower() or "full_eval" in raw:
        category = "eval"
    else:
        category = "other"
    categories[category] += rss
    counts[category] += 1

memory = {}
with open("/proc/meminfo") as handle:
    for line in handle:
        key, value, *_ = line.split()
        if key.rstrip(":") in ("MemTotal", "MemAvailable"):
            memory[key.rstrip(":")] = int(value) * 1024
cpu = 100 * float(open("/proc/loadavg").read().split()[0]) / (os.cpu_count() or 1)
io = 0.0
try:
    for token in open("/proc/pressure/io").readline().split():
        if token.startswith("avg10="):
            io = float(token.split("=")[1])
except OSError:
    pass
root = os.statvfs("/")
fields = [
    f"cpu={cpu:.1f}", f"io={io:.2f}",
    f"totalGiB={memory.get('MemTotal', 0) / 2**30:.1f}",
    f"availGiB={memory.get('MemAvailable', 0) / 2**30:.1f}",
    f"projectRssGiB={sum(categories.values()) / 2**30:.1f}",
    f"rootFreeGiB={root.f_bavail * root.f_frsize / 2**30:.1f}",
]
fields.extend(f"{key}={categories[key] / 2**30:.1f}G/{counts[key]}p" for key in categories)
print(" ".join(fields))
