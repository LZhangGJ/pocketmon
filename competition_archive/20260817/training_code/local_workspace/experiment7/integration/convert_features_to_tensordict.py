from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_from_pocketmon_replays import write_tensordict_memmap_store


def convert(source: Path, output: Path) -> dict:
    source = source.resolve()
    output = output.resolve()
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    receipt = write_tensordict_memmap_store(output, arrays)
    receipt.update({"source": str(source), "tensorCount": len(arrays)})
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert compressed Experiment 7 features to a TensorDict-style memmap store"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = convert(args.source, args.output)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
