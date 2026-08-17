from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "experiment7" / "integration"
TRAINING = ROOT / "experiment7" / "reference" / "training"
for path in (INTEGRATION, TRAINING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_from_pocketmon_replays import write_tensordict_memmap_store


def test_tensordict_memmap_store_round_trip(tmp_path: Path) -> None:
    arrays = {
        "state_features": np.arange(12, dtype=np.float32).reshape(3, 4),
        "option_features": np.arange(20, dtype=np.float32).reshape(5, 4),
        "option_labels": np.asarray([1, 0, 0, 1, 0], dtype=np.uint8),
        "option_offsets": np.asarray([0, 2, 3, 5], dtype=np.int64),
        "episode_ids": np.asarray([10, 11, 12], dtype=np.int64),
        "min_counts": np.zeros(3, dtype=np.int8),
        "max_counts": np.ones(3, dtype=np.int8),
        "chosen_counts": np.ones(3, dtype=np.int8),
        "validation": np.asarray([0, 0, 1], dtype=np.uint8),
        "is_winners": np.asarray([1, 0, 1], dtype=np.uint8),
        "policy_weights": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
    }
    store = tmp_path / "features_tensordict"
    receipt = write_tensordict_memmap_store(store, arrays)

    metadata = json.loads((store / "meta.json").read_text(encoding="utf-8"))
    assert receipt["kind"] == "experiment7_tensordict_memmap"
    assert metadata["kind"] == "experiment7_tensordict_memmap"
    for name, expected in arrays.items():
        actual = np.load(store / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        assert isinstance(actual, np.memmap)
        np.testing.assert_array_equal(actual, expected)


def test_bundle_loads_npz_and_tensordict_identically(tmp_path: Path) -> None:
    from train import Bundle
    from feature_tensor_store import install_memmap_bundle_loader

    install_memmap_bundle_loader(Bundle)

    arrays = {
        "state_features": np.arange(12, dtype=np.float32).reshape(3, 4),
        "option_features": np.arange(20, dtype=np.float32).reshape(5, 4),
        "option_labels": np.asarray([1, 0, 0, 1, 0], dtype=np.uint8),
        "option_offsets": np.asarray([0, 2, 3, 5], dtype=np.int64),
        "episode_ids": np.asarray([10, 11, 12], dtype=np.int64),
        "min_counts": np.zeros(3, dtype=np.int8),
        "max_counts": np.ones(3, dtype=np.int8),
        "chosen_counts": np.ones(3, dtype=np.int8),
        "validation": np.asarray([0, 0, 1], dtype=np.uint8),
        "is_winners": np.asarray([1, 0, 1], dtype=np.uint8),
        "policy_weights": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
    }
    npz = tmp_path / "features.npz"
    np.savez_compressed(npz, **arrays)
    store = tmp_path / "features_tensordict"
    write_tensordict_memmap_store(store, arrays)

    token_cache = tmp_path / "token_cache"
    token_cache.mkdir()
    np.save(token_cache / "entity_cat.npy", np.zeros((3, 1, 1), dtype=np.int32))
    np.save(token_cache / "entity_num.npy", np.zeros((3, 1, 1), dtype=np.float32))
    np.save(token_cache / "entity_mask.npy", np.ones((3, 1), dtype=np.uint8))
    np.save(token_cache / "semantic_hash.npy", np.zeros((3, 2), dtype=np.uint64))
    np.save(token_cache / "semantic_labels.npy", np.zeros((3, 2), dtype=np.uint8))

    compressed = Bundle.load(npz, token_cache)
    mapped = Bundle.load(store, token_cache)
    assert isinstance(mapped.data["state_features"], np.memmap)
    assert mapped.data.keys() == compressed.data.keys()
    for name in compressed.data:
        np.testing.assert_array_equal(mapped.data[name], compressed.data[name])


def test_materialize_row_migrates_all_training_caches(tmp_path: Path) -> None:
    from materialize_tensordict_sources import materialize_row

    source = tmp_path / "source"
    source.mkdir()
    arrays = {
        "state_features": np.zeros((2, 4), dtype=np.float32),
        "option_features": np.zeros((2, 4), dtype=np.float32),
        "option_labels": np.asarray([1, 1], dtype=np.uint8),
        "option_offsets": np.asarray([0, 1, 2], dtype=np.int64),
        "episode_ids": np.asarray([1, 2], dtype=np.int64),
        "min_counts": np.zeros(2, dtype=np.int8),
        "max_counts": np.ones(2, dtype=np.int8),
        "chosen_counts": np.ones(2, dtype=np.int8),
        "validation": np.asarray([0, 1], dtype=np.uint8),
        "is_winners": np.ones(2, dtype=np.uint8),
        "policy_weights": np.ones(2, dtype=np.float32),
    }
    features = source / "features.npz"
    np.savez_compressed(features, **arrays)
    required = {
        "token_cache": ("entity_cat.npy", "entity_num.npy", "entity_mask.npy", "semantic_hash.npy", "semantic_labels.npy"),
        "sequence_cache": ("history_indices.npy", "expert_action_features.npy"),
        "identity_cache": ("own_deck_cards.npy", "opponent_deck_labels.npy", "opponent_visible_unique_cards.npy"),
    }
    for directory, names in required.items():
        root = source / directory
        root.mkdir()
        for name in names:
            np.save(root / name, np.zeros((2, 1), dtype=np.float32))
    decisions = source / "decisions.jsonl.gz"
    decisions.write_bytes(b"placeholder")
    row = {
        "name": "2026-08-12",
        "features": str(features),
        "decisions": str(decisions),
        "tokenCache": str(source / "token_cache"),
        "sequenceCache": str(source / "sequence_cache"),
        "identityCache": str(source / "identity_cache"),
    }
    updated, receipt = materialize_row(row, tmp_path / "migrated")
    assert receipt["parity"]["passed"] is True
    assert Path(updated["features"]).is_dir()
    for key in ("tokenCache", "sequenceCache", "identityCache"):
        assert Path(updated[key]).is_dir()
        assert receipt["migratedCaches"][key]["status"] == "copied"


def test_atomic_directory_copy_keeps_concurrent_winner(tmp_path: Path) -> None:
    import materialize_tensordict_sources

    source = tmp_path / "source"
    source.mkdir()
    (source / "payload.txt").write_text("source", encoding="utf-8")
    target = tmp_path / "target"

    def concurrent_replace(_temporary: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "payload.txt").write_text("winner", encoding="utf-8")
        raise FileExistsError(destination)

    with patch.object(materialize_tensordict_sources.os, "replace", concurrent_replace):
        status = materialize_tensordict_sources.copy_directory_atomic(source, target)

    assert status == "existing_after_race"
    assert (target / "payload.txt").read_text(encoding="utf-8") == "winner"
    assert not list(tmp_path.glob(".target-*"))
