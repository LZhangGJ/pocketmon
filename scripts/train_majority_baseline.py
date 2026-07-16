from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data" / "processed" / "baseline_train_rows.csv"
MODEL_PATH = ROOT / "models" / "majority_baseline.json"

FEATURE_COLUMNS = [
    "select_type",
    "select_context",
    "option_count",
    "min_count",
    "max_count",
]


def train_majority_mapping(df: pd.DataFrame) -> dict:
    grouped = (
        df.groupby(FEATURE_COLUMNS + ["target_action"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    mapping: dict[str, str] = {}
    for _, row in grouped.iterrows():
        key = "|".join(str(row[c]) for c in FEATURE_COLUMNS)
        if key not in mapping:
            mapping[key] = row["target_action"]

    global_default = (
        df["target_action"].value_counts().idxmax() if not df.empty else "[]"
    )

    return {
        "feature_columns": FEATURE_COLUMNS,
        "mapping": mapping,
        "global_default": global_default,
        "train_rows": int(len(df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a majority-vote baseline from prepared replay rows")
    parser.add_argument("--input", default=str(INPUT_CSV))
    parser.add_argument("--output", default=str(MODEL_PATH))
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df = pd.read_csv(in_path)
    model = train_majority_mapping(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")

    print(f"train_rows={model['train_rows']}")
    print(f"mapping_size={len(model['mapping'])}")
    print(f"output={out_path}")


if __name__ == "__main__":
    main()
