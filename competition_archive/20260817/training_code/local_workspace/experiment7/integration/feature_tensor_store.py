from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


KIND = "experiment7_tensordict_memmap"


def load_memmap_arrays(features_path: Path) -> dict[str, np.ndarray]:
    metadata_path = features_path / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("kind") != KIND:
        raise RuntimeError(f"unsupported feature tensor store: {metadata_path}")
    tensors = metadata.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise RuntimeError(f"empty feature tensor store: {metadata_path}")
    data: dict[str, np.ndarray] = {}
    for name, receipt in tensors.items():
        path = features_path / str(receipt["path"])
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != receipt.get("shape"):
            raise RuntimeError(f"tensor shape mismatch: {path}")
        if array.dtype.str != receipt.get("dtype"):
            raise RuntimeError(f"tensor dtype mismatch: {path}")
        data[name] = array
    return data


def install_memmap_bundle_loader(bundle_class: type[Any]) -> None:
    """Extend the vendored Bundle loader without modifying the frozen reference."""

    if getattr(bundle_class, "_experiment7_memmap_loader", False):
        return
    original_load = bundle_class.load.__func__

    @classmethod
    def load(cls, features_path: Path, cache_dir: Path):
        if not features_path.is_dir():
            return original_load(cls, features_path, cache_dir)
        data = load_memmap_arrays(features_path)
        result = cls(
            features_path=features_path,
            cache_dir=cache_dir,
            data=data,
            entity_cat=np.load(cache_dir / "entity_cat.npy", mmap_mode="r"),
            entity_num=np.load(cache_dir / "entity_num.npy", mmap_mode="r"),
            entity_mask=np.load(cache_dir / "entity_mask.npy", mmap_mode="r"),
            semantic_hash=np.load(cache_dir / "semantic_hash.npy", mmap_mode="r"),
            semantic_labels=np.load(cache_dir / "semantic_labels.npy", mmap_mode="r"),
        )
        decisions = len(data["episode_ids"])
        if any(
            array.shape[0] != decisions
            for array in (
                result.entity_cat,
                result.entity_num,
                result.entity_mask,
                result.semantic_hash,
                result.semantic_labels,
            )
        ):
            raise RuntimeError(f"cache decision count mismatch for {features_path}")
        return result

    bundle_class.load = load
    bundle_class._experiment7_memmap_loader = True
