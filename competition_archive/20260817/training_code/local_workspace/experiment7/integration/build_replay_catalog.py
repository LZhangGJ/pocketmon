from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from common import (
    Experiment7Error,
    canonical_deck,
    canonical_deck_sha256,
    counter_dict,
    is_forced_decision,
    parse_timestamp,
    sha256_file,
    utc_now,
    validate_action,
    write_csv,
    write_deck,
    write_json,
)


CATALOG_FIELDS = [
    "episode_id",
    "create_time",
    "create_timestamp",
    "raw_path",
    "json_bytes",
    "json_sha256",
    "source_date",
    "module_version",
    "is_clean",
    "status0",
    "status1",
    "winner_index",
    "team0",
    "team1",
    "reward0",
    "reward1",
    "min_score",
    "avg_score",
    "sum_score",
    "deck0_sha256",
    "deck1_sha256",
    "policy_weight0",
    "policy_weight1",
    "aligned_decisions0",
    "aligned_decisions1",
    "nonforced_decisions0",
    "nonforced_decisions1",
    "policy_decisions0",
    "policy_decisions1",
    "forced_decisions0",
    "forced_decisions1",
    "invalid_actions0",
    "invalid_actions1",
    "source_archive",
    "source_archive_sha256",
]


class ZipReplaySource:
    """A replay member read directly from one already-open ZIP archive."""

    def __init__(
        self,
        archive_path: Path,
        archive: zipfile.ZipFile,
        member: str,
        *,
        source_date: str,
        mtime: float,
    ) -> None:
        self.archive_path = archive_path
        self.archive = archive
        self.member = member
        self.source_date = source_date
        self.mtime = mtime

    @property
    def stem(self) -> str:
        return Path(self.member).stem

    @property
    def locator(self) -> str:
        return f"zip://{self.archive_path.resolve()}#{self.member}"

    def read_bytes(self) -> bytes:
        return self.archive.read(self.member)

    def __str__(self) -> str:
        return self.locator


def _as_number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _manifest_metadata(
    root: Path, archive: zipfile.ZipFile | None = None
) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    if archive is not None:
        manifest_name = next(
            (
                name
                for name in archive.namelist()
                if Path(name).name.lower() == "manifest.csv"
            ),
            None,
        )
        if manifest_name is None:
            return rows
        with archive.open(manifest_name) as raw_handle:
            handle = io.TextIOWrapper(raw_handle, encoding="utf-8-sig", newline="")
            for row in csv.DictReader(handle):
                raw_episode = row.get("episode_id") or row.get("id")
                if raw_episode is not None:
                    rows[int(raw_episode)] = dict(row)
        return rows
    for manifest in sorted(root.rglob("manifest.csv")):
        if any(part.startswith(".") for part in manifest.parts):
            continue
        try:
            with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    raw_episode = row.get("episode_id") or row.get("id")
                    if raw_episode is None:
                        continue
                    try:
                        episode = int(raw_episode)
                    except ValueError:
                        continue
                    existing = rows.get(episode)
                    if existing is None or (not existing.get("create_time") and row.get("create_time")):
                        rows[episode] = dict(row)
        except (OSError, UnicodeError, csv.Error):
            continue
    return rows


def _episode_id(path: Path | ZipReplaySource, payload: dict[str, Any]) -> int:
    try:
        return int(path.stem)
    except ValueError:
        value = payload.get("episode_id") or payload.get("id")
        if isinstance(value, int):
            return value
        raise Experiment7Error(f"cannot infer numeric episode ID from {path}")


def _extract_teams(payload: dict[str, Any]) -> list[str]:
    info = payload.get("info") or {}
    names = info.get("TeamNames")
    if isinstance(names, list):
        return [str(value) for value in names[:2]]
    agents = info.get("Agents")
    if isinstance(agents, list):
        return [str(item.get("Name", "")) if isinstance(item, dict) else "" for item in agents[:2]]
    return []


def _extract_decks(
    payload: dict[str, Any], source: Path | ZipReplaySource
) -> list[tuple[int, ...]]:
    decks: list[tuple[int, ...] | None] = [None, None]
    for pair in (payload.get("steps") or [])[:12]:
        if not isinstance(pair, list):
            continue
        for player, step in enumerate(pair[:2]):
            if decks[player] is not None or not isinstance(step, dict):
                continue
            action = step.get("action")
            if (
                isinstance(action, list)
                and len(action) == 60
                and all(isinstance(card, int) and not isinstance(card, bool) for card in action)
            ):
                decks[player] = canonical_deck(action)
        if all(value is not None for value in decks):
            break
    if any(value is None for value in decks):
        raise Experiment7Error(f"{source}: could not recover both 60-card decks")
    return [value for value in decks if value is not None]


def _winner_index(rewards: list[Any]) -> int:
    if len(rewards) < 2:
        return -1
    left = _as_number(rewards[0], float("nan"))
    right = _as_number(rewards[1], float("nan"))
    if left > right:
        return 0
    if right > left:
        return 1
    return 2


def _aligned_counts(payload: dict[str, Any], policy_weights: list[float]) -> list[dict[str, int]]:
    totals = [
        {"aligned": 0, "nonforced": 0, "policy": 0, "forced": 0, "invalid": 0}
        for _ in range(2)
    ]
    steps = payload.get("steps") or []
    for action_step in range(1, len(steps)):
        previous_pair = steps[action_step - 1]
        current_pair = steps[action_step]
        if not isinstance(previous_pair, list) or not isinstance(current_pair, list):
            continue
        for player in (0, 1):
            if player >= len(previous_pair) or player >= len(current_pair):
                continue
            previous = previous_pair[player]
            current = current_pair[player]
            if not isinstance(previous, dict) or not isinstance(current, dict):
                continue
            observation = previous.get("observation") or {}
            select = observation.get("select") if isinstance(observation, dict) else None
            if previous.get("status") != "ACTIVE" or not isinstance(select, dict):
                continue
            options = select.get("option") or []
            minimum = int(select.get("minCount", 0) or 0)
            maximum = int(select.get("maxCount", 0) or 0)
            valid, _ = validate_action(current.get("action"), options, minimum, maximum)
            if not valid:
                totals[player]["invalid"] += 1
                continue
            totals[player]["aligned"] += 1
            if is_forced_decision(options, minimum, maximum):
                totals[player]["forced"] += 1
            else:
                totals[player]["nonforced"] += 1
                if policy_weights[player] > 0:
                    totals[player]["policy"] += 1
    return totals


def _discover_json(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".json":
        return [root]
    paths = []
    for path in root.rglob("*.json"):
        relative = path.relative_to(root)
        if any(part.startswith("_") or part.startswith(".") for part in relative.parts[:-1]):
            continue
        paths.append(path)
    return sorted(paths)


def build_catalog(
    raw_root: Path,
    output_dir: Path,
    policy_source: str,
    strict: bool,
    max_files: int,
) -> dict[str, Any]:
    archive: zipfile.ZipFile | None = None
    if raw_root.is_file() and raw_root.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(raw_root)
        paths: list[Path | ZipReplaySource] = []
        for info in archive.infolist():
            if info.is_dir() or Path(info.filename).suffix.lower() != ".json":
                continue
            paths.append(
                ZipReplaySource(
                    raw_root,
                    archive,
                    info.filename,
                    source_date=raw_root.parent.name,
                    mtime=datetime(*info.date_time).timestamp(),
                )
            )
        paths.sort(key=lambda source: source.member if isinstance(source, ZipReplaySource) else str(source))
    else:
        paths = _discover_json(raw_root)
    metadata = _manifest_metadata(raw_root, archive)
    if max_files > 0:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"no replay JSON files under {raw_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    deck_dir = output_dir / "decks"
    deck_dir.mkdir(parents=True, exist_ok=True)
    catalog_rows: list[dict[str, Any]] = []
    unique_decks: dict[str, tuple[int, ...]] = {}
    duplicate_receipts: dict[int, str] = {}
    failures: list[dict[str, Any]] = []
    module_counts: Counter[str] = Counter()
    clean_module_counts: Counter[str] = Counter()

    for index, path in enumerate(paths, start=1):
        raw = path.read_bytes()
        raw_sha = hashlib.sha256(raw).hexdigest()
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise Experiment7Error("replay root is not an object")
            episode = _episode_id(path, payload)
            if episode in duplicate_receipts:
                if duplicate_receipts[episode] != raw_sha:
                    raise Experiment7Error(
                        f"conflicting duplicate episode {episode}: {duplicate_receipts[episode]} != {raw_sha}"
                    )
                continue
            duplicate_receipts[episode] = raw_sha
            decks = _extract_decks(payload, path)
            deck_hashes = [canonical_deck_sha256(deck) for deck in decks]
            for deck_hash, deck in zip(deck_hashes, decks):
                unique_decks.setdefault(deck_hash, deck)
            rewards = list((payload.get("rewards") or [])[:2])
            statuses = [str(value) for value in (payload.get("statuses") or [])[:2]]
            winner = _winner_index(rewards)
            if policy_source == "winners":
                policy_weights = [float(winner == player) for player in (0, 1)]
            elif policy_source == "nonlosers":
                policy_weights = [float(winner in (player, 2)) for player in (0, 1)]
            elif policy_source == "both":
                policy_weights = [1.0, 1.0]
            else:
                raise ValueError(policy_source)
            counts = _aligned_counts(payload, policy_weights)
            meta = metadata.get(episode, {})
            create_time = str(meta.get("create_time") or "")
            fallback_mtime = path.mtime if isinstance(path, ZipReplaySource) else path.stat().st_mtime
            timestamp = parse_timestamp(create_time, fallback_mtime)
            module = str(payload.get("module_version") or payload.get("moduleVersion") or "")
            module_counts[module] += 1
            valid_rewards = (
                len(rewards) >= 2
                and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in rewards)
                and winner in (0, 1, 2)
            )
            statuses_done = len(statuses) >= 2 and all(value == "DONE" for value in statuses)
            invalid = counts[0]["invalid"] + counts[1]["invalid"]
            is_clean = valid_rewards and statuses_done and invalid == 0 and bool(module)
            if is_clean:
                clean_module_counts[module] += 1
            teams = _extract_teams(payload)
            row = {
                "episode_id": episode,
                "create_time": create_time,
                "create_timestamp": f"{timestamp:.6f}",
                "raw_path": path.locator if isinstance(path, ZipReplaySource) else str(path.resolve()),
                "json_bytes": len(raw),
                "json_sha256": raw_sha,
                "source_date": path.source_date if isinstance(path, ZipReplaySource) else path.parent.name,
                "module_version": module,
                "is_clean": int(is_clean),
                "status0": statuses[0] if len(statuses) > 0 else "",
                "status1": statuses[1] if len(statuses) > 1 else "",
                "winner_index": winner,
                "team0": teams[0] if len(teams) > 0 else "",
                "team1": teams[1] if len(teams) > 1 else "",
                "reward0": rewards[0] if len(rewards) > 0 else "",
                "reward1": rewards[1] if len(rewards) > 1 else "",
                "min_score": meta.get("min_score", 0),
                "avg_score": meta.get("avg_score", 0),
                "sum_score": meta.get("sum_score", 0),
                "deck0_sha256": deck_hashes[0],
                "deck1_sha256": deck_hashes[1],
                "policy_weight0": policy_weights[0],
                "policy_weight1": policy_weights[1],
                "aligned_decisions0": counts[0]["aligned"],
                "aligned_decisions1": counts[1]["aligned"],
                "nonforced_decisions0": counts[0]["nonforced"],
                "nonforced_decisions1": counts[1]["nonforced"],
                "policy_decisions0": counts[0]["policy"],
                "policy_decisions1": counts[1]["policy"],
                "forced_decisions0": counts[0]["forced"],
                "forced_decisions1": counts[1]["forced"],
                "invalid_actions0": counts[0]["invalid"],
                "invalid_actions1": counts[1]["invalid"],
                "source_archive": str(path.archive_path.resolve()) if isinstance(path, ZipReplaySource) else "",
                "source_archive_sha256": "",
            }
            catalog_rows.append(row)
        except Exception as exc:
            failure = {
                "path": str(path),
                "sha256": raw_sha,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            if strict:
                raise Experiment7Error(json.dumps(failure, ensure_ascii=False)) from exc
        if index == 1 or index % 250 == 0 or index == len(paths):
            print(json.dumps({"processed": index, "total": len(paths), "failures": len(failures)}), flush=True)

    catalog_rows.sort(key=lambda row: (float(row["create_timestamp"]), int(row["episode_id"])))
    if archive is not None:
        archive.close()
    catalog_path = output_dir / "replay_catalog.csv"
    write_csv(catalog_path, catalog_rows, CATALOG_FIELDS)

    deck_map: dict[str, str] = {}
    for deck_hash, deck in sorted(unique_decks.items()):
        path = write_deck(deck_dir / f"{deck_hash}.csv", deck)
        deck_map[deck_hash] = str(path.resolve())
    deck_map_path = output_dir / "deck_map.json"
    write_json(deck_map_path, {"schemaVersion": 1, "hashAlgorithm": "sha256(canonical sorted JSON deck multiset)", "decks": deck_map})

    clean_rows = [row for row in catalog_rows if int(row["is_clean"]) == 1]
    payload = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "rawRoot": str(raw_root.resolve()),
        "policySource": policy_source,
        "strict": strict,
        "jsonFiles": len(paths),
        "catalogEpisodes": len(catalog_rows),
        "cleanEpisodes": len(clean_rows),
        "failedFiles": len(failures),
        "duplicateIdenticalEpisodes": len(paths) - len(catalog_rows) - len(failures),
        "uniqueDecks": len(unique_decks),
        "moduleCounts": counter_dict(module for module in module_counts.elements()),
        "cleanModuleCounts": counter_dict(module for module in clean_module_counts.elements()),
        "totalPolicyDecisions": int(
            sum(int(row["policy_decisions0"]) + int(row["policy_decisions1"]) for row in clean_rows)
        ),
        "totalNonforcedDecisions": int(
            sum(int(row["nonforced_decisions0"]) + int(row["nonforced_decisions1"]) for row in clean_rows)
        ),
        "invalidAlignedActions": int(
            sum(int(row["invalid_actions0"]) + int(row["invalid_actions1"]) for row in catalog_rows)
        ),
        "catalog": {"path": str(catalog_path.resolve()), "sha256": sha256_file(catalog_path)},
        "deckMap": {"path": str(deck_map_path.resolve()), "sha256": sha256_file(deck_map_path)},
        "failures": failures[:100],
    }
    write_json(output_dir / "catalog_receipt.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a direct raw-replay catalog for Experiment 7")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-source", choices=("winners", "nonlosers", "both"), default="winners")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()
    build_catalog(
        args.raw_root.resolve(),
        args.output_dir.resolve(),
        args.policy_source,
        args.strict,
        args.max_files,
    )


if __name__ == "__main__":
    main()
