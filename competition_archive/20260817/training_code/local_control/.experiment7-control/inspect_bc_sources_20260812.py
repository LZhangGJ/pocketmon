import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
wanted = set(sys.argv[2:])
for row in payload.get("datasets", []):
    if not wanted or row.get("name") in wanted:
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))
