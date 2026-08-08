from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tokenizer import (
    ENTITY_CAT_DIM,
    ENTITY_NUM_DIM,
    MAX_ENTITIES,
    encode_entities,
    expand_semantic_labels,
    load_cards,
    semantic_option_hash,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-entities", type=int, default=MAX_ENTITIES)
    parser.add_argument("--progress-every", type=int, default=10_000)
    args = parser.parse_args()

    cards = load_cards(args.catalog)
    with np.load(args.features, mmap_mode="r") as archive:
        offsets = archive["option_offsets"].copy()
        original_labels = archive["option_labels"].copy()
        decision_count = len(archive["episode_ids"])
    max_actions = int(np.max(offsets[1:] - offsets[:-1]))
    if max_actions > 64:
        raise RuntimeError(f"unexpected max action count {max_actions}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    entity_cat = np.lib.format.open_memmap(
        args.output_dir / "entity_cat.npy",
        mode="w+",
        dtype=np.int16,
        shape=(decision_count, args.max_entities, ENTITY_CAT_DIM),
    )
    entity_num = np.lib.format.open_memmap(
        args.output_dir / "entity_num.npy",
        mode="w+",
        dtype=np.float16,
        shape=(decision_count, args.max_entities, ENTITY_NUM_DIM),
    )
    entity_mask = np.lib.format.open_memmap(
        args.output_dir / "entity_mask.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(decision_count, args.max_entities),
    )
    semantic_hash = np.lib.format.open_memmap(
        args.output_dir / "semantic_hash.npy",
        mode="w+",
        dtype=np.uint32,
        shape=(decision_count, max_actions),
    )
    semantic_labels = np.lib.format.open_memmap(
        args.output_dir / "semantic_labels.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(decision_count, max_actions),
    )

    semantic_hash[:] = 0
    semantic_labels[:] = 0
    truncated_decisions = 0
    truncated_entities = 0
    entity_counts: list[int] = []
    seen = 0
    with gzip.open(args.decisions, "rt", encoding="utf-8") as handle:
        for line in handle:
            record: dict[str, Any] = json.loads(line)
            decision = int(record["decisionId"])
            if decision != seen:
                raise RuntimeError(f"decision order mismatch: expected {seen}, got {decision}")
            observation = record["observation"]
            options = record["legalOptions"]
            begin = int(offsets[decision])
            end = int(offsets[decision + 1])
            if len(options) != end - begin:
                raise RuntimeError(f"option count mismatch at decision {decision}")
            if options != (observation.get("select") or {}).get("option"):
                raise RuntimeError(f"raw/select option mismatch at decision {decision}")
            chosen = [int(index) for index in record["expertSelection"]]
            if any(index < 0 or index >= len(options) for index in chosen):
                raise RuntimeError(f"illegal expert selection at decision {decision}")

            cat, num, mask, truncated = encode_entities(observation, cards, args.max_entities)
            entity_cat[decision] = cat
            entity_num[decision] = num.astype(np.float16)
            entity_mask[decision] = mask
            count = int(mask.sum())
            entity_counts.append(count)
            if truncated:
                truncated_decisions += 1
                truncated_entities += truncated

            hashes = np.asarray(
                [semantic_option_hash(observation, option) for option in options],
                dtype=np.uint32,
            )
            labels = original_labels[begin:end]
            expected = np.zeros(len(options), dtype=np.uint8)
            expected[chosen] = 1
            if not np.array_equal(labels, expected):
                raise RuntimeError(f"expert label mismatch at decision {decision}")
            semantic_hash[decision, : len(options)] = hashes
            semantic_labels[decision, : len(options)] = expand_semantic_labels(labels, hashes).astype(np.uint8)
            seen += 1
            if args.progress_every and seen % args.progress_every == 0:
                print(json.dumps({"decisions": seen, "truncatedDecisions": truncated_decisions}), flush=True)

    if seen != decision_count:
        raise RuntimeError(f"raw decision count mismatch: expected {decision_count}, got {seen}")
    for array in (entity_cat, entity_num, entity_mask, semantic_hash, semantic_labels):
        array.flush()

    counts = np.asarray(entity_counts, dtype=np.int32)
    artifacts = {}
    for name in (
        "entity_cat.npy",
        "entity_num.npy",
        "entity_mask.npy",
        "semantic_hash.npy",
        "semantic_labels.npy",
    ):
        path = args.output_dir / name
        artifacts[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "schemaVersion": 1,
        "decisionCount": decision_count,
        "maxActions": max_actions,
        "maxEntities": args.max_entities,
        "entityCatDim": ENTITY_CAT_DIM,
        "entityNumDim": ENTITY_NUM_DIM,
        "maxObservedEntitiesAfterCap": int(counts.max(initial=0)),
        "p95EntitiesAfterCap": float(np.quantile(counts, 0.95)) if len(counts) else 0.0,
        "truncatedDecisions": truncated_decisions,
        "truncatedEntities": truncated_entities,
        "source": {
            "decisions": str(args.decisions.resolve()),
            "decisionsSha256": sha256(args.decisions),
            "features": str(args.features.resolve()),
            "featuresSha256": sha256(args.features),
            "catalog": str(args.catalog.resolve()),
            "catalogSha256": sha256(args.catalog),
        },
        "artifacts": artifacts,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
