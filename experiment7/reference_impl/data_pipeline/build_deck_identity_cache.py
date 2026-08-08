from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {int(row["episode_id"]): row for row in csv.DictReader(handle)}


def load_deck_map(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    decks: dict[str, np.ndarray] = {}
    receipts: dict[str, object] = {}
    for deck_hash, raw_path in payload["decks"].items():
        deck_path = Path(raw_path)
        values = np.asarray(
            [int(line) for line in deck_path.read_text(encoding="utf-8").splitlines() if line.strip()],
            dtype=np.int16,
        )
        if len(values) != 60 or np.any(values <= 0):
            raise RuntimeError(f"deck must contain 60 positive card IDs: {deck_path}")
        key = deck_hash.lower()
        decks[key] = values
        receipts[key] = {
            "path": str(deck_path.resolve()),
            "sha256": sha256(deck_path),
            "uniqueCards": int(len(np.unique(values))),
        }
    return decks, receipts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--token-cache", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--deck-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.features) as archive:
        episode_ids = archive["episode_ids"].copy()
        player_indices = archive["player_indices"].copy()
    decisions_count = len(episode_ids)
    entity_cat = np.load(args.token_cache / "entity_cat.npy", mmap_mode="r")
    entity_mask = np.load(args.token_cache / "entity_mask.npy", mmap_mode="r")
    if entity_cat.shape[0] != decisions_count or entity_mask.shape[0] != decisions_count:
        raise RuntimeError("token cache decision count mismatch")

    class_payload = json.loads(args.class_map.read_text(encoding="utf-8"))
    class_by_hash = {
        row["deckSha256"].lower(): int(row["index"])
        for row in class_payload["classes"]
        if row.get("deckSha256")
    }
    other_hashes = set(value.lower() for value in class_payload["other"]["deckHashes"])
    decks, deck_receipts = load_deck_map(args.deck_map)
    catalog = load_catalog(args.catalog)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    own_cards_path = args.output_dir / "own_deck_cards.npy"
    labels_path = args.output_dir / "opponent_deck_labels.npy"
    visible_path = args.output_dir / "opponent_visible_unique_cards.npy"
    own_cards = np.lib.format.open_memmap(
        own_cards_path, mode="w+", dtype=np.int16, shape=(decisions_count, 60)
    )
    labels = np.lib.format.open_memmap(
        labels_path, mode="w+", dtype=np.int16, shape=(decisions_count,)
    )
    visible = np.lib.format.open_memmap(
        visible_path, mode="w+", dtype=np.int16, shape=(decisions_count,)
    )
    labels[:] = -1
    label_counts: Counter[int] = Counter()
    own_counts: Counter[str] = Counter()
    missing_opponent_hash = 0
    seen = 0
    with gzip.open(args.decisions, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            decision = int(record["decisionId"])
            if decision != seen:
                raise RuntimeError(f"decision order mismatch: expected {seen}, got {decision}")
            episode = int(record["episodeId"])
            player = int(record["playerIndex"])
            if episode != int(episode_ids[decision]) or player != int(player_indices[decision]):
                raise RuntimeError(f"features/raw identity mismatch at decision {decision}")
            own_hash = str(record["deckSha256"]).lower()
            if own_hash not in decks:
                raise RuntimeError(f"own deck hash {own_hash} has no exact deck mapping")
            own_cards[decision] = decks[own_hash]
            own_counts[own_hash] += 1

            row = catalog.get(episode)
            if row is None:
                raise RuntimeError(f"episode {episode} missing from catalog")
            opponent_hash = row.get(f"deck{1 - player}_sha256", "").strip().lower()
            if not opponent_hash:
                missing_opponent_hash += 1
                label = -1
            elif opponent_hash in class_by_hash:
                label = class_by_hash[opponent_hash]
            elif opponent_hash in other_hashes:
                label = 0
            else:
                # A known but class-map-unseen hash is OTHER only outside class-map construction.
                label = 0
            labels[decision] = label
            label_counts[label] += 1

            mask = np.asarray(entity_mask[decision], dtype=bool)
            opponent = mask & (np.asarray(entity_cat[decision, :, 2]) == 1)
            ids = np.asarray(entity_cat[decision, opponent, 0], dtype=np.int32)
            visible[decision] = len(np.unique(ids[ids > 0]))
            seen += 1

    if seen != decisions_count:
        raise RuntimeError(f"raw decision count mismatch: expected {decisions_count}, got {seen}")
    for array in (own_cards, labels, visible):
        array.flush()
    del own_cards, labels, visible

    visible_read = np.load(visible_path, mmap_mode="r")
    labels_read = np.load(labels_path, mmap_mode="r")
    valid = labels_read >= 0
    payload = {
        "schemaVersion": 1,
        "privacyBoundary": (
            "opponent evidence is derived only from token-cache entities whose relative side is opponent; "
            "hidden hand/deck/prize cards are absent"
        ),
        "decisions": decisions_count,
        "deckSize": 60,
        "opponentClasses": len(class_payload["classes"]),
        "ownDeckDecisionCounts": dict(own_counts),
        "labelDecisionCounts": {str(key): value for key, value in sorted(label_counts.items())},
        "missingOpponentHashDecisions": missing_opponent_hash,
        "labeledDecisions": int(valid.sum()),
        "labeledWithVisibleEvidence": int(np.sum(valid & (visible_read > 0))),
        "noVisibleEvidenceDecisions": int(np.sum(visible_read == 0)),
        "deckReceipts": deck_receipts,
        "sources": {
            "decisions": {"path": str(args.decisions.resolve()), "sha256": sha256(args.decisions)},
            "features": {"path": str(args.features.resolve()), "sha256": sha256(args.features)},
            "tokenCacheManifest": {
                "path": str((args.token_cache / "manifest.json").resolve()),
                "sha256": sha256(args.token_cache / "manifest.json"),
            },
            "catalog": {"path": str(args.catalog.resolve()), "sha256": sha256(args.catalog)},
            "classMap": {"path": str(args.class_map.resolve()), "sha256": sha256(args.class_map)},
            "deckMap": {"path": str(args.deck_map.resolve()), "sha256": sha256(args.deck_map)},
        },
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (own_cards_path, labels_path, visible_path)
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
