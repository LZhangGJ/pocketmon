#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = args.reference_root.resolve()
    for required in (
        reference / "training" / "train.py",
        reference / "training" / "train_multideck_identity.py",
        reference / "data_pipeline" / "features.py",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    payload = json.loads(args.source.resolve().read_text(encoding="utf-8"))
    payload["referenceRoot"] = str(reference)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "referenceRoot": str(reference)}))


if __name__ == "__main__":
    main()
