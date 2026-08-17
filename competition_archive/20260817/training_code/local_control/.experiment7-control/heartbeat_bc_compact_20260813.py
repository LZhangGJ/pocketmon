import glob
import json
import os
import sys


roots = sys.argv[1:]
out = {}
for root in roots:
    for variant in ("standard_1m", "large_256x6"):
        path = os.path.join(root, variant, "training_report.json")
        if not os.path.isfile(path):
            continue
        d = json.load(open(path, encoding="utf-8"))
        epochs = d.get("epochs") or d.get("history") or []
        latest = epochs[-1] if epochs else d.get("latestEpoch") or d
        training = latest.get("training") or {}
        validation = latest.get("validation") or {}
        out[f"{os.path.basename(root)}:{variant}"] = {
            "epoch": latest.get("epoch") or len(epochs),
            "dps": training.get("decisionsPerSecond"),
            "trainNll": training.get("policyNll"),
            "semantic": validation.get("exactSemantic"),
            "index": validation.get("exactIndex"),
            "illegal": validation.get("illegalPredictionCount"),
            "portable": os.path.isfile(os.path.join(root, variant, "universal_bc.npz")),
            "parity": os.path.isfile(os.path.join(root, variant, "PARITY_PASSED")),
        }
print(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
