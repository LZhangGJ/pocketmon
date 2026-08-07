from __future__ import annotations

import argparse
import json
from pathlib import Path

FEATURE_COLUMNS = ["select_type", "select_context", "option_count", "min_count", "max_count"]


def featurize(select: dict) -> str:
    values = {
        "select_type": select.get("type", -1),
        "select_context": select.get("context", -1),
        "option_count": len(select.get("option", [])) if isinstance(select.get("option"), list) else 0,
        "min_count": select.get("minCount", -1),
        "max_count": select.get("maxCount", -1),
    }
    return "|".join(str(values[column]) for column in FEATURE_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a diagnostic majority-table prediction")
    parser.add_argument("--model", required=True)
    parser.add_argument("--select-json", default=None)
    parser.add_argument("--select-file", default=None)
    args = parser.parse_args()
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    if model.get("schema_version") != 2 or model.get("label_alignment") != "previous":
        raise RuntimeError("Legacy majority model has unvalidated replay alignment; retrain it with DATA-001/002 code")
    if args.select_file:
        select = json.loads(Path(args.select_file).read_text(encoding="utf-8-sig"))
    elif args.select_json:
        select = json.loads(args.select_json)
    else:
        raise ValueError("Provide either --select-json or --select-file")
    key = featurize(select)
    if key not in model["mapping"]:
        raise KeyError("Unseen selection signature; use the rule agent fallback instead of an unsafe default action")
    print(json.dumps(json.loads(model["mapping"][key]), ensure_ascii=False))


if __name__ == "__main__":
    main()
