import json
import sys

source, output = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    payload = json.load(handle)
rows = payload.get("datasets") or [payload["dataset"]]
payload.pop("dataset", None)
payload["datasets"] = [rows[0]]
with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, separators=(",", ":"))
print(rows[0].get("name", "first"))
