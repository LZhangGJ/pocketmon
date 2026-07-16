from __future__ import annotations

import argparse
import json
from pathlib import Path

FEATURE_COLUMNS = [
    "select_type",
    "select_context",
    "option_count",
    "min_count",
    "max_count",
]


def featurize(select: dict) -> str:
    values = {
        "select_type": select.get("type", -1),
        "select_context": select.get("context", -1),
        "option_count": len(select.get("option", [])) if isinstance(select.get("option"), list) else 0,
        "min_count": select.get("minCount", -1),
        "max_count": select.get("maxCount", -1),
    }
    return "|".join(str(values[c]) for c in FEATURE_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict action from select dict using trained majority baseline")
    parser.add_argument("--model", required=True)
    parser.add_argument("--select-json", default=None, help="JSON string of observation.select")
    parser.add_argument("--select-file", default=None, help="Path to JSON file containing observation.select")
    args = parser.parse_args()

    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    if args.select_file:
        select = json.loads(Path(args.select_file).read_text(encoding="utf-8-sig"))
    elif args.select_json:
        select = json.loads(args.select_json)
    else:
        raise ValueError("Provide either --select-json or --select-file")

    key = featurize(select)
    pred_text = model["mapping"].get(key, model["global_default"])
    prediction = json.loads(pred_text)
    print(json.dumps(prediction, ensure_ascii=False))


if __name__ == "__main__":
    main()
