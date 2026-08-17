#!/usr/bin/env bash
set -u
hosts=(10.113.13.53 10.113.13.54 10.113.13.57 10.113.13.63 10.113.13.64 10.113.13.67 10.113.13.68 10.113.13.69 10.113.13.71 10.113.13.72 10.113.13.73 10.113.13.74 10.113.13.75 10.113.13.77 10.113.13.78)
for host in "${hosts[@]}"; do
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "bash --noprofile --norc -c 'python3 - <<'\''PY'\''
import glob,os
for p in glob.glob("/proc/[0-9]*/cmdline"):
    try: args=open(p,"rb").read().decode(errors="replace").split("\\0")
    except OSError: continue
    cmd=" ".join(args)
    kind=None
    if "run_async_ppo_learner.py" in cmd: kind="PPO_LEARNER"
    elif "train_universal_ppo.py" in cmd: kind="PPO_UPDATE"
    elif "train_universal_bc.py" in cmd: kind="BC_TRAIN"
    if not kind: continue
    def val(flag):
        try:return args[args.index(flag)+1]
        except (ValueError,IndexError):return "-"
    target=val("--chain") if kind=="PPO_LEARNER" else val("--output") if kind=="PPO_UPDATE" else val("--output-dir")
    print(kind,os.path.basename(os.path.dirname(target)) if kind=="PPO_UPDATE" else target,val("--generation"),val("--device"),os.path.basename(val("--sources")))
PY' 2>/dev/null" | sed "s#^#$host #" || true
done
