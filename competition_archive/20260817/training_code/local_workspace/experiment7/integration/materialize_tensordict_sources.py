from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from build_from_pocketmon_replays import write_tensordict_memmap_store
from common import utc_now, write_json


def dataset_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("datasets")
    if rows is None:
        rows = [payload["dataset"]]
    if not isinstance(rows, list) or not rows:
        raise ValueError("source manifest has no dataset rows")
    return [dict(row) for row in rows]


def validate_store(store: Path) -> dict[str, Any]:
    metadata_path = store / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("kind") != "experiment7_tensordict_memmap":
        raise ValueError(f"unsupported tensor store: {metadata_path}")
    tensors = metadata.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError(f"empty tensor store: {metadata_path}")
    for name, receipt in tensors.items():
        path = store / str(receipt["path"])
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != receipt.get("shape"):
            raise ValueError(f"shape mismatch: {path}")
        if array.dtype.str != receipt.get("dtype"):
            raise ValueError(f"dtype mismatch: {path}")
    return metadata


def sample_indices(size: int) -> np.ndarray:
    if size <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(
        np.asarray([0, size // 7, size // 3, size // 2, (6 * size) // 7, size - 1])
    )


def parity_npz_to_store(source: Path, store: Path) -> dict[str, Any]:
    metadata = validate_store(store)
    tensors = metadata["tensors"]
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != set(tensors):
            raise ValueError(
                f"tensor key mismatch: source={sorted(archive.files)} "
                f"store={sorted(tensors)}"
            )
        checked = 0
        for name in archive.files:
            expected = archive[name]
            actual = np.load(
                store / str(tensors[name]["path"]), mmap_mode="r", allow_pickle=False
            )
            if expected.shape != actual.shape or expected.dtype != actual.dtype:
                raise ValueError(f"tensor metadata mismatch: {name}")
            if expected.ndim == 0:
                np.testing.assert_array_equal(expected, actual)
            else:
                indices = sample_indices(expected.shape[0])
                np.testing.assert_array_equal(expected[indices], actual[indices])
            checked += 1
    return {"passed": True, "tensorCount": checked, "sampleRule": "six_rows_per_tensor"}


def copy_directory_atomic(source: Path, target: Path) -> str:
    if source.resolve() == target.resolve():
        return "existing"
    if target.exists():
        return "existing"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    temporary.rmdir()
    try:
        shutil.copytree(source, temporary, copy_function=shutil.copy2)
        try:
            os.replace(temporary, target)
        except FileExistsError:
            # Two daily-window builders may finish the same shared cache copy at
            # nearly the same time.  The winner owns ``target``; the loser must
            # keep it intact and discard only its own private temporary tree.
            if not target.is_dir():
                raise
            shutil.rmtree(temporary)
            return "existing_after_race"
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return "copied"


def materialize_row(row: dict[str, Any], output_root: Path | None) -> tuple[dict[str, Any], dict[str, Any]]:
    original = Path(row.get("featuresNpz") or row["features"]).resolve()
    current = Path(row["features"]).resolve()
    name = str(row.get("name") or original.parent.name)
    dataset_root = output_root / name if output_root is not None else None
    if current.is_dir():
        store = dataset_root / "features_tensordict" if dataset_root is not None else current
        status = copy_directory_atomic(current, store)
        metadata = validate_store(store)
        parity = (
            parity_npz_to_store(original, store)
            if original.is_file()
            else {"passed": True, "tensorCount": len(metadata["tensors"]), "sampleRule": "source_store_metadata"}
        )
    else:
        if current.suffix != ".npz" or not original.is_file():
            raise ValueError(f"expected a compressed feature archive: {current}")
        store = dataset_root / "features_tensordict" if dataset_root is not None else original.with_name("features_tensordict")
        if store.exists():
            status = "existing"
        else:
            with np.load(original, allow_pickle=False) as archive:
                arrays = {tensor_name: archive[tensor_name] for tensor_name in archive.files}
            write_tensordict_memmap_store(store, arrays)
            status = "created"
        parity = parity_npz_to_store(original, store)
    updated = {**row, "features": str(store)}
    if original.is_file():
        updated["featuresNpz"] = str(original)
    migrated_caches: dict[str, dict[str, str]] = {}
    if dataset_root is not None:
        for key, directory_name in (
            ("tokenCache", "token_cache"),
            ("sequenceCache", "sequence_cache"),
            ("identityCache", "identity_cache"),
        ):
            source_cache = Path(row[key]).resolve()
            target_cache = dataset_root / directory_name
            cache_status = copy_directory_atomic(source_cache, target_cache)
            required = {
                "tokenCache": ("entity_cat.npy", "entity_num.npy", "entity_mask.npy", "semantic_hash.npy", "semantic_labels.npy"),
                "sequenceCache": ("history_indices.npy", "expert_action_features.npy"),
                "identityCache": ("own_deck_cards.npy", "opponent_deck_labels.npy", "opponent_visible_unique_cards.npy"),
            }[key]
            missing = [filename for filename in required if not (target_cache / filename).is_file()]
            if missing:
                raise ValueError(f"incomplete migrated cache: {target_cache} missing={missing}")
            updated[key] = str(target_cache)
            migrated_caches[key] = {
                "status": cache_status,
                "source": str(source_cache),
                "target": str(target_cache),
            }
    return updated, {
        "name": name,
        "status": status,
        "source": str(original if original.is_file() else current),
        "store": str(store),
        "parity": parity,
        "migratedCaches": migrated_caches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize every Universal BC feature shard as TensorDict-style memmaps"
    )
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store-root", type=Path)
    args = parser.parse_args()
    source_path = args.sources.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    rows = dataset_rows(payload)
    output_root = args.store_root.resolve() if args.store_root else None
    converted: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for row in rows:
        updated, receipt = materialize_row(row, output_root)
        converted.append(updated)
        receipts.append(receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    result = dict(payload)
    if "datasets" in result:
        result["datasets"] = converted
    else:
        result["dataset"] = converted[0]
    result["tensorStorage"] = {
        "kind": "experiment7_tensordict_memmap",
        "createdAt": utc_now(),
        "sourceManifest": str(source_path),
        "parity": "shape/dtype plus six deterministic rows per tensor",
        "datasets": receipts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    write_json(temporary, result)
    os.replace(temporary, output_path)
    print(json.dumps({"output": str(output_path), "datasets": len(converted)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
