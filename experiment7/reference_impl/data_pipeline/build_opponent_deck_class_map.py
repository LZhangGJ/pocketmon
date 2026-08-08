from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {int(row["episode_id"]): row for row in csv.DictReader(handle)}
    if not rows:
        raise RuntimeError(f"empty catalog: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        nargs=4,
        action="append",
        metavar=("NAME", "DECISIONS", "CATALOG", "CAL_EPISODES"),
        required=True,
    )
    parser.add_argument("--min-train-actors", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.min_train_actors <= 0:
        raise ValueError("min-train-actors must be positive")

    actor_keys: dict[str, set[tuple[str, int, int]]] = {}
    decision_counts: Counter[str] = Counter()
    source_receipts: list[dict[str, object]] = []
    for name, decisions_text, catalog_text, calibration_text in args.source:
        decisions = Path(decisions_text)
        catalog_path = Path(catalog_text)
        catalog = load_catalog(catalog_path)
        calibration_count = int(calibration_text)
        train_episode_order: list[int] = []
        train_episode_seen: set[int] = set()
        with gzip.open(decisions, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("split") != "train":
                    continue
                episode = int(record["episodeId"])
                if episode not in train_episode_seen:
                    train_episode_seen.add(episode)
                    train_episode_order.append(episode)
        if calibration_count <= 0 or calibration_count >= len(train_episode_order):
            raise RuntimeError(f"{name}: invalid calibration episode count")
        calibration_episodes = set(train_episode_order[-calibration_count:])
        seen = 0
        with gzip.open(decisions, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("split") != "train":
                    continue
                episode = int(record["episodeId"])
                if episode in calibration_episodes:
                    continue
                player = int(record["playerIndex"])
                row = catalog.get(episode)
                if row is None:
                    raise RuntimeError(f"episode {episode} missing from {catalog_path}")
                opponent_hash = row.get(f"deck{1 - player}_sha256", "").strip().lower()
                if not opponent_hash:
                    continue
                actor_keys.setdefault(opponent_hash, set()).add((name, episode, player))
                decision_counts[opponent_hash] += 1
                seen += 1
        source_receipts.append(
            {
                "name": name,
                "decisions": str(decisions.resolve()),
                "decisionsSha256": sha256(decisions),
                "catalog": str(catalog_path.resolve()),
                "catalogSha256": sha256(catalog_path),
                "labeledTrainDecisions": seen,
                "fitEpisodes": len(train_episode_order) - calibration_count,
                "excludedCalibrationEpisodes": calibration_count,
            }
        )

    selected = [
        deck_hash
        for deck_hash, keys in actor_keys.items()
        if len(keys) >= args.min_train_actors
    ]
    selected.sort(key=lambda deck_hash: (-len(actor_keys[deck_hash]), deck_hash))
    classes = [
        {"index": 0, "name": "OTHER", "deckSha256": None},
        *[
            {
                "index": index,
                "name": deck_hash[:12],
                "deckSha256": deck_hash,
                "trainActorEpisodes": len(actor_keys[deck_hash]),
                "trainDecisions": decision_counts[deck_hash],
            }
            for index, deck_hash in enumerate(selected, start=1)
        ],
    ]
    other_hashes = sorted(set(actor_keys) - set(selected))
    payload = {
        "schemaVersion": 1,
        "privacyBoundary": (
            "labels come from replay catalog only; runtime inputs contain actor-visible "
            "opponent cards only; unknown catalog hashes are ignored"
        ),
        "selectionBoundary": "fit episodes only; chronological calibration tails excluded",
        "minTrainActorEpisodes": args.min_train_actors,
        "classes": classes,
        "other": {
            "deckHashes": other_hashes,
            "trainActorEpisodes": sum(len(actor_keys[value]) for value in other_hashes),
            "trainDecisions": sum(decision_counts[value] for value in other_hashes),
        },
        "allKnownTrainDecks": len(actor_keys),
        "sources": source_receipts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
