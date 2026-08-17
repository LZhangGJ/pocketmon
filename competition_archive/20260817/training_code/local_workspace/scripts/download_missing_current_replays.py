from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
import requests


OUTPUT = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/current-leaderboard-scoregt1000")
EPISODES = Path("/homes/lzhang/pocketmon/.tmp_current_leaderboard/current_episode_index.json")
PROGRESS = OUTPUT / "monitoring/api_download_progress.json"
TARGETS = {
    "a02_baseline": {"cafa7652a6349be806d8ac2b9abfdb6c72ca3821f368e0d912e2d989f3b54cdd"},
    "a02_pokegear": {"a672e91d6d736dcfeee2dceee72c1d13793e04d79d41651f13ba814d8fbec634"},
    "a08_maximum_belt": {"1e9cb92d8b026b70281cc0d17f2218b0225d3e6225358334b9b2147c81af2b71"},
    "a08_rabsca": {"15e237d601341ee6ea5546aff74df3eab87ff77ae5055d02ba79079716770e0e"},
    "gold_exact_lucario": {"dc8571d0bc2e546a1f85b938696cfc40a1451c68a4ccc1f695e7c3e1c74f1278"},
    "dragapult_munkidori_interference": {"89e6155f25310ee695c0761c85d3ae8e44f376456ff0539231820f8e803f2d5e"},
}
ALL_DECKS = Path("/homes/lzhang/pocketmon/.tmp_current_leaderboard/all_decks")
RUNS = Path("/dataT0/Free/lzhang/pocketmon-runs")
TARGET_DECK_PATHS = {
    "a02_baseline": ALL_DECKS / "cafa7652a6349be806d8ac2b9abfdb6c72ca3821f368e0d912e2d989f3b54cdd.csv",
    "a02_pokegear": RUNS / "experiment7-async-ppo-league-20260811/control/ppo-reallocation-20260813/decks/a02_g247_pokegear.csv",
    "a08_maximum_belt": ALL_DECKS / "1e9cb92d8b026b70281cc0d17f2218b0225d3e6225358334b9b2147c81af2b71.csv",
    "a08_rabsca": RUNS / "experiment7-async-ppo-league-20260811/control/ppo-reallocation-20260813/decks/a08_rabsca.csv",
    "gold_exact_lucario": ALL_DECKS / "dc8571d0bc2e546a1f85b938696cfc40a1451c68a4ccc1f695e7c3e1c74f1278.csv",
    "dragapult_munkidori_interference": ALL_DECKS / "89e6155f25310ee695c0761c85d3ae8e44f376456ff0539231820f8e803f2d5e.csv",
}
LOCAL = threading.local()
_ORIGINAL_SEND = requests.Session.send


def _send_with_timeout(self, request, **kwargs):
    kwargs.setdefault("timeout", (10, 60))
    return _ORIGINAL_SEND(self, request, **kwargs)


requests.Session.send = _send_with_timeout


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def deck_hash(cards):
    if not isinstance(cards, list) or len(cards) != 60 or any(type(x) is not int for x in cards):
        raise ValueError("invalid deck")
    raw = json.dumps(tuple(sorted(cards)), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def replay_decks(payload):
    result = [None, None]
    cards_result = [None, None]
    for pair in (payload.get("steps") or [])[:12]:
        if not isinstance(pair, list):
            continue
        for side, step in enumerate(pair[:2]):
            if result[side] is None and isinstance(step, dict):
                action = step.get("action")
                if isinstance(action, list) and len(action) == 60:
                    result[side] = deck_hash(action)
                    cards_result[side] = action
        if all(result):
            return result, cards_result
    raise ValueError("both decks unavailable")


def load_deck(path):
    cards = [int(value) for value in path.read_text(encoding="utf-8").split()]
    if len(cards) != 60:
        raise ValueError(f"target deck is not 60 cards: {path}")
    return cards


def replacement_distance(left, right):
    from collections import Counter
    a, b = Counter(left), Counter(right)
    return sum(abs(a[key] - b[key]) for key in a.keys() | b.keys()) // 2


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    temp = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        os.link(source, temp)
    except OSError:
        shutil.copy2(source, temp)
    os.replace(temp, target)


def api():
    if not hasattr(LOCAL, "api"):
        LOCAL.api = KaggleApi()
        LOCAL.api.authenticate()
    return LOCAL.api


def download_one(episode_id, metadata, prototypes):
    stage = OUTPUT / "_api_staging" / str(episode_id)
    stage.mkdir(parents=True, exist_ok=True)
    replay = stage / f"episode-{episode_id}-replay.json"
    error = None
    try:
        for delay in (0, 2, 5, 10):
            if delay:
                time.sleep(delay)
            try:
                if not replay.is_file():
                    api().competition_episode_replay(int(episode_id), str(stage), quiet=True)
                payload = json.loads(replay.read_text(encoding="utf-8"))
                decks, cardlists = replay_decks(payload)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        else:
            return str(episode_id), "error", None, error
        matched = defaultdict(list)
        match_distance = defaultdict(dict)
        for side, cards in enumerate(cardlists):
            distances = {target: replacement_distance(cards, prototype) for target, prototype in prototypes.items()}
            best = min(distances.values())
            if best <= 3:
                for target, distance in distances.items():
                    if distance == best:
                        matched[target].append(side)
                        match_distance[target][str(side)] = distance
        if not matched:
            replay.unlink(missing_ok=True)
            return str(episode_id), "non_target", None, None
        digest = file_hash(replay)
        object_path = OUTPUT / "_objects/sha256" / f"{digest}.json"
        link(replay, object_path)
        date = metadata["create_time"][:10]
        for target in matched:
            link(object_path, OUTPUT / "by_deck" / target / date / f"{episode_id}.json")
        replay.unlink(missing_ok=True)
        row = {
            "episode_id": int(episode_id),
            "source_date": date,
            "create_time": metadata["create_time"],
            "qualifying_sources": metadata["qualifying_sources"],
            "agents": metadata["agents"],
            "deck_hashes": decks,
            "target_sides": dict(matched),
            "target_replacement_distance": dict(match_distance),
            "replay_sha256": digest,
            "object_path": str(object_path),
            "source_catalog": None,
            "source_api": "competition_episode_replay",
        }
        return str(episode_id), "target", row, None
    except Exception as exc:
        return str(episode_id), "error", None, f"{type(exc).__name__}: {exc}"


def main():
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN is not set")
    episode_index = json.loads(EPISODES.read_text(encoding="utf-8"))
    latest_path = OUTPUT / "monitoring/latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    manifest_path = OUTPUT / "monitoring/global_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {str(row["episode_id"]): row for row in manifest}
    progress = json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.is_file() else {"completed": {}, "errors": {}, "rows": {}}
    progress.setdefault("rows", {})
    rows.update(progress["rows"])
    prototypes = {name: load_deck(path) for name, path in TARGET_DECK_PATHS.items()}
    pending = [
        str(x)
        for x in latest["missing_replay_episode_ids"]
        if str(x) not in progress["completed"]
        or (progress["completed"].get(str(x)) == "target" and str(x) not in progress["rows"])
    ]
    done = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(download_one, episode_id, episode_index["episodes"][episode_id], prototypes): episode_id
            for episode_id in pending
        }
        for future in as_completed(futures):
            episode_id, status, row, error = future.result()
            done += 1
            if status == "error":
                progress["errors"][episode_id] = error
            else:
                progress["completed"][episode_id] = status
                progress["errors"].pop(episode_id, None)
            if row:
                rows[episode_id] = row
                progress["rows"][episode_id] = row
            if done % 20 == 0:
                atomic_json(PROGRESS, progress)
                print(json.dumps({"done_this_run": done, "pending_this_run": len(pending), "targets_total": sum(v == "target" for v in progress["completed"].values()), "errors": len(progress["errors"])}), flush=True)
    merged = sorted(rows.values(), key=lambda row: row["episode_id"])
    counts = defaultdict(int)
    for row in merged:
        for target in row["target_sides"]:
            counts[target] += 1
    latest["matching_replays"] = len(merged)
    latest["by_deck"] = {name: counts.get(name, 0) for name in TARGETS}
    latest["api_download"] = {
        "eligible_missing": len(latest["missing_replay_episode_ids"]),
        "completed": len(progress["completed"]),
        "target": sum(value == "target" for value in progress["completed"].values()),
        "non_target": sum(value == "non_target" for value in progress["completed"].values()),
        "errors": len(progress["errors"]),
    }
    atomic_json(PROGRESS, progress)
    atomic_json(manifest_path, merged)
    atomic_json(latest_path, latest)
    print(json.dumps({"matching_replays": len(merged), "by_deck": latest["by_deck"], "api_download": latest["api_download"]}))


if __name__ == "__main__":
    main()
