from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DATA = ROOT / "experiment7" / "reference_impl" / "data_pipeline"
if str(REFERENCE_DATA) not in sys.path:
    sys.path.insert(0, str(REFERENCE_DATA))

from features import OPTION_DIM, STATE_DIM, encode_option, encode_state, load_catalog  # noqa: E402

from common import (  # noqa: E402
    canonical_deck_sha256,
    first_present,
    open_jsonl,
    parse_float,
    parse_int,
    sha256_file,
    write_deck,
    write_json,
)


def _forced(select: dict[str, Any], option_count: int) -> bool:
    minimum = parse_int(select.get("minCount"), 0)
    maximum = parse_int(select.get("maxCount"), minimum)
    return minimum == maximum and (minimum == 0 or minimum == option_count)


def _episode_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Experiment 7 requires numeric episode IDs; got {value!r}") from exc


def _manifest_time(row: dict[str, Any]) -> str:
    manifest = row.get("manifest") or {}
    return str(first_present(manifest, ("create_time", "CreateTime", "created_at", "timestamp"), ""))


def _module_version(row: dict[str, Any]) -> str:
    manifest = row.get("manifest") or {}
    observation = row.get("observation") or {}
    current = observation.get("current") or {}
    return str(
        first_present(
            manifest,
            ("module_version", "moduleVersion", "ModuleVersion"),
            first_present(current, ("moduleVersion", "module_version"), "unknown"),
        )
    )


def _team_name(row: dict[str, Any]) -> str:
    manifest = row.get("manifest") or {}
    return str(first_present(manifest, ("team_name", "teamName", "TeamName"), ""))


@dataclass
class DatasetBuilder:
    name: str
    output_dir: Path
    cards: dict[int, dict[str, Any]]
    attacks: dict[int, dict[str, Any]]
    state_features: list[np.ndarray] = field(default_factory=list)
    option_features: list[np.ndarray] = field(default_factory=list)
    option_labels: list[int] = field(default_factory=list)
    option_offsets: list[int] = field(default_factory=lambda: [0])
    episode_ids: list[int] = field(default_factory=list)
    source_steps: list[int] = field(default_factory=list)
    player_indices: list[int] = field(default_factory=list)
    min_counts: list[int] = field(default_factory=list)
    max_counts: list[int] = field(default_factory=list)
    chosen_counts: list[int] = field(default_factory=list)
    select_types: list[int] = field(default_factory=list)
    select_contexts: list[int] = field(default_factory=list)
    turns: list[int] = field(default_factory=list)
    validation: list[int] = field(default_factory=list)
    deck_hashes: list[str] = field(default_factory=list)
    team_names: list[str] = field(default_factory=list)
    is_winners: list[int] = field(default_factory=list)
    policy_weights: list[float] = field(default_factory=list)
    raw_records: list[dict[str, Any]] = field(default_factory=list)

    def add(self, row: dict[str, Any], deck_hash: str, *, split: str) -> None:
        observation = row.get("observation") or {}
        select = observation.get("select") or {}
        options = select.get("option") or []
        action = row.get("action")
        if not isinstance(options, list) or not isinstance(action, list):
            raise ValueError("canonical row is missing legal options or action")
        minimum = parse_int(select.get("minCount"), 0)
        maximum = parse_int(select.get("maxCount"), minimum)
        if not (
            minimum <= len(action) <= maximum
            and len(action) == len(set(action))
            and all(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(options) for index in action)
        ):
            raise ValueError(
                f"invalid canonical action episode={row.get('episode_id')} player={row.get('player')}"
            )

        chosen = set(int(index) for index in action)
        self.state_features.append(encode_state(observation))
        for option_index, option in enumerate(options):
            self.option_features.append(
                encode_option(observation, option, option_index, self.cards, self.attacks)
            )
            self.option_labels.append(int(option_index in chosen))
        self.option_offsets.append(len(self.option_features))
        episode = _episode_int(row["episode_id"])
        player = int(row["player"])
        source_step = parse_int(row.get("observation_step"), parse_int(row.get("action_step"), 1) - 1)
        winner = int(float(row.get("outcome") or 0.0) > 0.0)
        policy_weight = float(row.get("policy_weight") or 0.0)
        self.episode_ids.append(episode)
        self.source_steps.append(source_step)
        self.player_indices.append(player)
        self.min_counts.append(minimum)
        self.max_counts.append(maximum)
        self.chosen_counts.append(len(action))
        self.select_types.append(parse_int(select.get("type"), -1))
        self.select_contexts.append(parse_int(select.get("context"), -1))
        self.turns.append(parse_int((observation.get("current") or {}).get("turn"), 0))
        self.validation.append(int(split == "validation"))
        self.deck_hashes.append(deck_hash)
        self.team_names.append(_team_name(row))
        self.is_winners.append(winner)
        self.policy_weights.append(policy_weight)
        self.raw_records.append(
            {
                "schemaVersion": 2,
                "decisionId": len(self.raw_records),
                "episodeId": episode,
                "sourceStep": source_step,
                "actionStep": parse_int(row.get("action_step"), source_step + 1),
                "playerIndex": player,
                "teamName": _team_name(row),
                "deckSha256": deck_hash,
                "split": split,
                "isWinner": bool(winner),
                "policyWeight": policy_weight,
                "moduleVersion": _module_version(row),
                "observation": observation,
                "legalOptions": options,
                "expertSelection": [int(value) for value in action],
            }
        )

    def finalize(self, extra_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.raw_records:
            raise RuntimeError(f"{self.name}: no decisions were selected")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = self.output_dir / "decisions.jsonl.gz"
        with gzip.open(raw_path, "wt", encoding="utf-8", newline="\n", compresslevel=6) as handle:
            for record in self.raw_records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        features_path = self.output_dir / "features.npz"
        np.savez_compressed(
            features_path,
            state_features=np.asarray(self.state_features, dtype=np.float32).reshape(-1, STATE_DIM),
            option_features=np.asarray(self.option_features, dtype=np.float32).reshape(-1, OPTION_DIM),
            option_labels=np.asarray(self.option_labels, dtype=np.uint8),
            option_offsets=np.asarray(self.option_offsets, dtype=np.int64),
            episode_ids=np.asarray(self.episode_ids, dtype=np.int64),
            source_steps=np.asarray(self.source_steps, dtype=np.int32),
            player_indices=np.asarray(self.player_indices, dtype=np.int8),
            min_counts=np.asarray(self.min_counts, dtype=np.int8),
            max_counts=np.asarray(self.max_counts, dtype=np.int8),
            chosen_counts=np.asarray(self.chosen_counts, dtype=np.int8),
            select_types=np.asarray(self.select_types, dtype=np.int8),
            select_contexts=np.asarray(self.select_contexts, dtype=np.int8),
            turns=np.asarray(self.turns, dtype=np.int16),
            validation=np.asarray(self.validation, dtype=np.uint8),
            deck_hashes=np.asarray(self.deck_hashes, dtype="U64"),
            team_names=np.asarray(self.team_names, dtype="U128"),
            is_winners=np.asarray(self.is_winners, dtype=np.uint8),
            policy_weights=np.asarray(self.policy_weights, dtype=np.float32),
        )
        decisions_csv = self.output_dir / "decisions.csv"
        with decisions_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            fields = [
                "decision_id", "episode_id", "source_step", "player_index", "split",
                "policy_weight", "turn", "select_type", "select_context", "option_count",
                "min_count", "max_count", "chosen_count",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, record in enumerate(self.raw_records):
                writer.writerow(
                    {
                        "decision_id": index,
                        "episode_id": record["episodeId"],
                        "source_step": record["sourceStep"],
                        "player_index": record["playerIndex"],
                        "split": record["split"],
                        "policy_weight": record["policyWeight"],
                        "turn": self.turns[index],
                        "select_type": self.select_types[index],
                        "select_context": self.select_contexts[index],
                        "option_count": self.option_offsets[index + 1] - self.option_offsets[index],
                        "min_count": self.min_counts[index],
                        "max_count": self.max_counts[index],
                        "chosen_count": self.chosen_counts[index],
                    }
                )
        summary = {
            "schemaVersion": 1,
            "name": self.name,
            "decisions": len(self.raw_records),
            "trainDecisions": int(len(self.raw_records) - sum(self.validation)),
            "validationDecisions": int(sum(self.validation)),
            "episodes": len(set(self.episode_ids)),
            "policyDecisions": int(sum(value > 0 for value in self.policy_weights)),
            "legalOptions": len(self.option_features),
            "expertSelections": int(sum(self.option_labels)),
            "stateFeatureDimension": STATE_DIM,
            "optionFeatureDimension": OPTION_DIM,
            "artifacts": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in (raw_path, features_path, decisions_csv)
            },
            **(extra_summary or {}),
        }
        write_json(self.output_dir / "summary.json", summary)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Experiment 7 feature datasets from pocketmon's audited canonical replay JSONL"
    )
    parser.add_argument("--canonical-decisions", type=Path, required=True)
    parser.add_argument("--deck-sidecar", type=Path, required=True)
    parser.add_argument("--selected-decks", type=Path, required=True)
    parser.add_argument("--engine-catalog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pretrain-max-decisions", type=int, default=0)
    args = parser.parse_args()

    selected_payload = json.loads(args.selected_decks.read_text(encoding="utf-8"))
    selected = selected_payload["selected"]
    if not selected:
        raise ValueError("selected deck manifest is empty")
    selected_by_hash = {row["deck_sha256"]: row for row in selected}
    holdout_fraction = float(selected_payload["thresholds"]["holdout_fraction"])

    actor_decks: dict[tuple[str, int], tuple[str, list[int]]] = {}
    episode_decks: dict[str, dict[int, str]] = defaultdict(dict)
    deck_cards: dict[str, list[int]] = {}
    for row in open_jsonl(args.deck_sidecar):
        episode = str(row["episode_id"])
        player = int(row["player"])
        cards = [int(value) for value in row["deck"]]
        deck_hash = canonical_deck_sha256(cards)
        key = (episode, player)
        previous = actor_decks.setdefault(key, (deck_hash, cards))
        if previous[0] != deck_hash:
            raise RuntimeError(f"conflicting deck sidecar for {key}")
        episode_decks[episode][player] = deck_hash
        deck_cards.setdefault(deck_hash, cards)

    selected_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    episode_meta: dict[str, dict[str, Any]] = {}
    broad_rows: list[tuple[dict[str, Any], str]] = []
    for row in open_jsonl(args.canonical_decisions):
        key = (str(row["episode_id"]), int(row["player"]))
        own = actor_decks.get(key)
        if own is None:
            raise RuntimeError(f"missing own deck for canonical row {key}")
        deck_hash = own[0]
        meta = episode_meta.setdefault(
            key[0],
            {
                "episode_id": key[0],
                "create_time": _manifest_time(row),
                "module_version": _module_version(row),
                "winner_index": row.get("winner"),
                "policy_weight0": 0.0,
                "policy_weight1": 0.0,
                "team0": "",
                "team1": "",
            },
        )
        player = key[1]
        meta[f"policy_weight{player}"] = max(
            float(meta[f"policy_weight{player}"]), float(row.get("policy_weight") or 0.0)
        )
        if _team_name(row):
            meta[f"team{player}"] = _team_name(row)
        observation = row.get("observation") or {}
        select = observation.get("select") or {}
        options = select.get("option") or []
        if float(row.get("policy_weight") or 0.0) > 0.0 and not _forced(select, len(options)):
            if not args.pretrain_max_decisions or len(broad_rows) < args.pretrain_max_decisions:
                broad_rows.append((row, deck_hash))
        if deck_hash in selected_by_hash and float(row.get("policy_weight") or 0.0) > 0.0:
            selected_rows[deck_hash].append(row)

    args.output_root.mkdir(parents=True, exist_ok=True)
    registry = args.output_root / "deck_registry"
    deck_map_payload: dict[str, Any] = {"schemaVersion": 1, "decks": {}}
    for deck_hash, cards in sorted(deck_cards.items()):
        path = registry / f"{deck_hash}.csv"
        write_deck(path, cards)
        deck_map_payload["decks"][deck_hash] = str(path.resolve())
    deck_map_path = args.output_root / "deck_map.json"
    write_json(deck_map_path, deck_map_payload)

    catalog_path = args.output_root / "catalog.csv"
    catalog_fields = [
        "episode_id", "is_clean", "module_version", "deck0_sha256", "deck1_sha256",
        "policy_weight0", "policy_weight1", "team0", "team1", "winner_index",
        "create_time", "min_score", "avg_score",
    ]
    with catalog_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=catalog_fields)
        writer.writeheader()
        for episode, meta in sorted(
            episode_meta.items(), key=lambda item: (item[1].get("create_time") or "", _episode_int(item[0]))
        ):
            decks = episode_decks.get(episode, {})
            if 0 not in decks or 1 not in decks:
                continue
            writer.writerow(
                {
                    "episode_id": _episode_int(episode),
                    "is_clean": 1,
                    "module_version": meta["module_version"],
                    "deck0_sha256": decks[0],
                    "deck1_sha256": decks[1],
                    "policy_weight0": meta["policy_weight0"],
                    "policy_weight1": meta["policy_weight1"],
                    "team0": meta["team0"],
                    "team1": meta["team1"],
                    "winner_index": meta["winner_index"] if meta["winner_index"] is not None else -1,
                    "create_time": meta["create_time"],
                    "min_score": "",
                    "avg_score": "",
                }
            )

    cards, attacks = load_catalog(args.engine_catalog)
    pretrain_builder = DatasetBuilder("pretrain", args.output_root / "pretrain", cards, attacks)
    for row, deck_hash in broad_rows:
        pretrain_builder.add(row, deck_hash, split="train")
    pretrain_summary = pretrain_builder.finalize(
        {"selection": "positive-policy nonforced decisions across all exact decks"}
    )

    current_sources = []
    for manifest_row in selected:
        deck_hash = manifest_row["deck_sha256"]
        rows = selected_rows.get(deck_hash, [])
        actor_episodes: dict[int, str] = {}
        for row in rows:
            episode = _episode_int(row["episode_id"])
            actor_episodes.setdefault(episode, _manifest_time(row))
        ordered_episodes = [
            episode
            for episode, _ in sorted(actor_episodes.items(), key=lambda item: (item[1], item[0]))
        ]
        holdout_count = max(1, math.ceil(len(ordered_episodes) * holdout_fraction))
        if holdout_count >= len(ordered_episodes):
            raise RuntimeError(f"{manifest_row['name']}: holdout leaves no train episodes")
        holdout = set(ordered_episodes[-holdout_count:])
        rows.sort(
            key=lambda row: (
                actor_episodes[_episode_int(row["episode_id"])],
                _episode_int(row["episode_id"]),
                int(row["player"]),
                parse_int(row.get("action_step"), 0),
            )
        )
        builder = DatasetBuilder(
            manifest_row["name"], args.output_root / "current" / manifest_row["name"], cards, attacks
        )
        for row in rows:
            builder.add(
                row,
                deck_hash,
                split="validation" if _episode_int(row["episode_id"]) in holdout else "train",
            )
        summary = builder.finalize(
            {
                "deckSha256": deck_hash,
                "archetypeLabel": manifest_row["archetype_label"],
                "calibrationEpisodeCount": int(manifest_row["calibration_episode_count"]),
                "chronologicalHoldoutEpisodes": len(holdout),
            }
        )
        current_sources.append(
            {
                **manifest_row,
                "dataset_dir": str(builder.output_dir.resolve()),
                "features": str((builder.output_dir / "features.npz").resolve()),
                "decisions": str((builder.output_dir / "decisions.jsonl.gz").resolve()),
                "summary": summary,
            }
        )

    manifest = {
        "schema_version": 1,
        "canonical_decisions": {"path": str(args.canonical_decisions.resolve()), "sha256": sha256_file(args.canonical_decisions)},
        "deck_sidecar": {"path": str(args.deck_sidecar.resolve()), "sha256": sha256_file(args.deck_sidecar)},
        "engine_catalog": {"path": str(args.engine_catalog.resolve()), "sha256": sha256_file(args.engine_catalog)},
        "catalog": {"path": str(catalog_path.resolve()), "sha256": sha256_file(catalog_path)},
        "deck_map": {"path": str(deck_map_path.resolve()), "sha256": sha256_file(deck_map_path)},
        "pretrain": {
            "dataset_dir": str((args.output_root / "pretrain").resolve()),
            "features": str((args.output_root / "pretrain" / "features.npz").resolve()),
            "decisions": str((args.output_root / "pretrain" / "decisions.jsonl.gz").resolve()),
            "summary": pretrain_summary,
        },
        "current_sources": current_sources,
    }
    write_json(args.output_root / "dataset_manifest.json", manifest)
    print(args.output_root / "dataset_manifest.json")


if __name__ == "__main__":
    main()
