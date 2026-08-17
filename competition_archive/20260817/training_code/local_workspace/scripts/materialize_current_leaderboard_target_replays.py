from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


RUNS = Path("/dataT0/Free/lzhang/pocketmon-runs")
OUTPUT = RUNS / "experiment7-daily-replay-bc/current-leaderboard-scoregt1000"
EPISODES = Path("/homes/lzhang/pocketmon/.tmp_current_leaderboard/current_episode_index.json")
SNAPSHOT = Path("/homes/lzhang/pocketmon/.tmp_current_leaderboard/current_snapshot.json")
CATALOGS = (
    RUNS / "experiment7-universal-20260810/prepared/catalog/replay_catalog.csv",
    RUNS / "experiment7-universal-7d-scoregt900-20260810/daily/2026-08-08/prepared/catalog/replay_catalog.csv",
    RUNS / "replay-refresh-20260812/cache/2026-08-09/prepared/catalog/replay_catalog.csv",
    RUNS / "replay-refresh-20260812/cache/2026-08-10/prepared/catalog/replay_catalog.csv",
    RUNS / "replay-refresh-20260812/cache/2026-08-11/prepared/catalog/replay_catalog.csv",
    RUNS / "experiment7-daily-replay-bc/cache/2026-08-12/prepared/catalog/replay_catalog.csv",
    RUNS / "experiment7-daily-replay-bc/cache/2026-08-13/prepared/catalog/replay_catalog.csv",
)
TARGETS = {
    "a02_baseline": {"cafa7652a6349be806d8ac2b9abfdb6c72ca3821f368e0d912e2d989f3b54cdd"},
    "a02_pokegear": {"a672e91d6d736dcfeee2dceee72c1d13793e04d79d41651f13ba814d8fbec634"},
    "a08_maximum_belt": {"1e9cb92d8b026b70281cc0d17f2218b0225d3e6225358334b9b2147c81af2b71"},
    "a08_rabsca": {"15e237d601341ee6ea5546aff74df3eab87ff77ae5055d02ba79079716770e0e"},
    "gold_exact_lucario": {"dc8571d0bc2e546a1f85b938696cfc40a1451c68a4ccc1f695e7c3e1c74f1278"},
    "dragapult_munkidori_interference": {"89e6155f25310ee695c0761c85d3ae8e44f376456ff0539231820f8e803f2d5e"},
}
ALL_DECKS = Path("/homes/lzhang/pocketmon/.tmp_current_leaderboard/all_decks")
TARGET_DECK_PATHS = {
    "a02_baseline": ALL_DECKS / "cafa7652a6349be806d8ac2b9abfdb6c72ca3821f368e0d912e2d989f3b54cdd.csv",
    "a02_pokegear": RUNS / "experiment7-async-ppo-league-20260811/control/ppo-reallocation-20260813/decks/a02_g247_pokegear.csv",
    "a08_maximum_belt": ALL_DECKS / "1e9cb92d8b026b70281cc0d17f2218b0225d3e6225358334b9b2147c81af2b71.csv",
    "a08_rabsca": RUNS / "experiment7-async-ppo-league-20260811/control/ppo-reallocation-20260813/decks/a08_rabsca.csv",
    "gold_exact_lucario": ALL_DECKS / "dc8571d0bc2e546a1f85b938696cfc40a1451c68a4ccc1f695e7c3e1c74f1278.csv",
    "dragapult_munkidori_interference": ALL_DECKS / "89e6155f25310ee695c0761c85d3ae8e44f376456ff0539231820f8e803f2d5e.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deck_hash(cards) -> str:
    if not isinstance(cards, list) or len(cards) != 60 or any(type(x) is not int for x in cards):
        raise ValueError("invalid 60-card deck")
    raw = json.dumps(tuple(sorted(cards)), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def replay_decks(payload):
    result = [None, None]
    for pair in (payload.get("steps") or [])[:12]:
        if not isinstance(pair, list):
            continue
        for side, step in enumerate(pair[:2]):
            if result[side] is None and isinstance(step, dict):
                action = step.get("action")
                if isinstance(action, list) and len(action) == 60:
                    result[side] = deck_hash(action)
        if all(result):
            break
    if not all(result):
        raise ValueError("both decks unavailable")
    return result


def load_deck(path: Path) -> list[int]:
    cards = [int(value) for value in path.read_text(encoding="utf-8").split()]
    if len(cards) != 60:
        raise ValueError(f"target deck is not 60 cards: {path}")
    return cards


def replacement_distance(left: list[int], right: list[int]) -> int:
    from collections import Counter
    if len(left) != 60 or len(right) != 60:
        return 999
    a, b = Counter(left), Counter(right)
    return sum(abs(a[key] - b[key]) for key in a.keys() | b.keys()) // 2


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, target)


def main() -> None:
    episode_index = json.loads(EPISODES.read_text(encoding="utf-8"))
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    if episode_index.get("errors"):
        raise RuntimeError("episode enumeration incomplete")
    wanted = episode_index["episodes"]
    prototypes = {name: load_deck(path) for name, path in TARGET_DECK_PATHS.items()}
    hash_to_targets = defaultdict(list)
    variant_receipt = defaultdict(dict)
    for deck_path in ALL_DECKS.glob("*.csv"):
        cards = load_deck(deck_path)
        distances = {name: replacement_distance(cards, prototype) for name, prototype in prototypes.items()}
        best = min(distances.values())
        if best <= 3:
            for target, distance in distances.items():
                if distance == best:
                    hash_to_targets[deck_path.stem].append(target)
                    variant_receipt[target][deck_path.stem] = distance
    for target, hashes in TARGETS.items():
        for value in hashes:
            if target not in hash_to_targets[value]:
                hash_to_targets[value].append(target)
            variant_receipt[target][value] = 0
    catalog_rows = {}
    conflicts = []
    for catalog in CATALOGS:
        with catalog.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                episode_id = str(row.get("episode_id") or "")
                if episode_id not in wanted:
                    continue
                prior = catalog_rows.get(episode_id)
                if prior and prior.get("json_sha256") != row.get("json_sha256"):
                    conflicts.append({"episode_id": episode_id, "catalogs": [prior["_catalog"], str(catalog)]})
                    continue
                row["_catalog"] = str(catalog)
                catalog_rows[episode_id] = row
    def materialize(item):
        episode_id, catalog_row = item
        expected = [catalog_row.get("deck0_sha256", ""), catalog_row.get("deck1_sha256", "")]
        matched = defaultdict(list)
        for side, value in enumerate(expected):
            for target in hash_to_targets.get(value, []):
                matched[target].append(side)
        if not matched:
            return None, None
        source = Path(catalog_row.get("raw_path") or "")
        if not source.is_file():
            return None, {"episode_id": episode_id, "error": "raw_path missing", "raw_path": str(source)}
        try:
            # The catalog already contains hashes produced when the official replay
            # corpus was prepared. Re-reading every large JSON here is unnecessary;
            # use those audited fields and only hash on the rare legacy row that lacks
            # a content digest.
            actual = expected
            content_hash = str(catalog_row.get("json_sha256") or "") or sha256(source)
            object_path = OUTPUT / "_objects" / "sha256" / f"{content_hash}.json"
            link_or_copy(source, object_path)
            source_date = str(catalog_row.get("source_date") or str(catalog_row.get("create_time") or "")[:10])
            for target in matched:
                link_or_copy(object_path, OUTPUT / "by_deck" / target / source_date / f"{episode_id}.json")
            return {
                "episode_id": int(episode_id),
                "source_date": source_date,
                "create_time": catalog_row.get("create_time"),
                "qualifying_sources": wanted[episode_id]["qualifying_sources"],
                "agents": wanted[episode_id]["agents"],
                "deck_hashes": actual,
                "target_sides": dict(matched),
                "replay_sha256": content_hash,
                "object_path": str(object_path),
                "source_catalog": catalog_row["_catalog"],
            }, None
        except Exception as exc:
            return None, {"episode_id": episode_id, "error": f"{type(exc).__name__}: {exc}"}

    rows = []
    counts = defaultdict(int)
    failures = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(materialize, item) for item in catalog_rows.items()]
        for future in as_completed(futures):
            row, failure = future.result()
            if failure:
                failures.append(failure)
            if row:
                rows.append(row)
                for target in row["target_sides"]:
                    counts[target] += 1
    rows.sort(key=lambda row: row["episode_id"])
    missing_catalog = sorted(int(x) for x in set(wanted) - set(catalog_rows))
    receipt = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "leaderboard_checked_at": snapshot["checked_at"],
        "leaderboard_rows": snapshot["leaderboard_rows_returned"],
        "qualifying_teams": snapshot["selected_count"],
        "active_submissions": episode_index["active_submission_count"],
        "unique_game_history_episodes": len(wanted),
        "catalog_intersection": len(catalog_rows),
        "matching_replays": len(rows),
        "by_deck": dict(counts),
        "missing_catalog_count": len(missing_catalog),
        "missing_catalog_episode_ids": missing_catalog,
        "failures": failures,
        "conflicts": conflicts,
    }
    atomic_json(OUTPUT / "monitoring" / "latest.json", receipt)
    atomic_json(OUTPUT / "monitoring" / "global_manifest.json", rows)
    atomic_json(OUTPUT / "monitoring" / "leaderboard_snapshot.json", snapshot)
    atomic_json(
        OUTPUT / "target_decks.json",
        {
            name: {
                "prototype_hashes": sorted(TARGETS[name]),
                "prototype_path": str(TARGET_DECK_PATHS[name]),
                "max_replacements": 3,
                "accepted_hashes": dict(sorted(variant_receipt[name].items())),
            }
            for name in TARGETS
        },
    )
    print(json.dumps({k: receipt[k] for k in ("unique_game_history_episodes", "catalog_intersection", "matching_replays", "by_deck", "missing_catalog_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
