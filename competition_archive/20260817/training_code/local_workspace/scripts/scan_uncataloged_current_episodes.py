from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


RAW = Path("/dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays")
OUTPUT = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-daily-replay-bc/current-leaderboard-scoregt1000")
EPISODES = Path("/homes/lzhang/pocketmon/.tmp_current_leaderboard/current_episode_index.json")
TARGETS = {
    "a02_baseline": {"cafa7652a6349be806d8ac2b9abfdb6c72ca3821f368e0d912e2d989f3b54cdd"},
    "a02_pokegear": {"a672e91d6d736dcfeee2dceee72c1d13793e04d79d41651f13ba814d8fbec634"},
    "a08_maximum_belt": {"1e9cb92d8b026b70281cc0d17f2218b0225d3e6225358334b9b2147c81af2b71"},
    "a08_rabsca": {"15e237d601341ee6ea5546aff74df3eab87ff77ae5055d02ba79079716770e0e"},
    "gold_exact_lucario": {"dc8571d0bc2e546a1f85b938696cfc40a1451c68a4ccc1f695e7c3e1c74f1278"},
    "dragapult_munkidori_interference": {"89e6155f25310ee695c0761c85d3ae8e44f376456ff0539231820f8e803f2d5e"},
}


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
    for pair in (payload.get("steps") or [])[:12]:
        if not isinstance(pair, list):
            continue
        for side, step in enumerate(pair[:2]):
            if result[side] is None and isinstance(step, dict):
                action = step.get("action")
                if isinstance(action, list) and len(action) == 60:
                    result[side] = deck_hash(action)
        if all(result):
            return result
    raise ValueError("both decks unavailable")


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
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        os.link(source, temp)
    except OSError:
        shutil.copy2(source, temp)
    os.replace(temp, target)


def main():
    episode_index = json.loads(EPISODES.read_text(encoding="utf-8"))
    latest_path = OUTPUT / "monitoring/latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    manifest_path = OUTPUT / "monitoring/global_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing = {str(row["episode_id"]): row for row in manifest}
    staging = list((RAW / "_staging").glob("*"))
    hash_to_targets = defaultdict(list)
    for target, hashes in TARGETS.items():
        for value in hashes:
            hash_to_targets[value].append(target)

    def locate(episode_id, date):
        candidates = [RAW / date / f"{episode_id}.json"]
        candidates.extend(root / date / f"{episode_id}.json" for root in staging)
        return next((path for path in candidates if path.is_file()), None)

    def inspect(episode_id):
        metadata = episode_index["episodes"][episode_id]
        date = metadata["create_time"][:10]
        source = locate(episode_id, date)
        if source is None:
            return episode_id, None, "not_found"
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            decks = replay_decks(payload)
            matched = defaultdict(list)
            for side, value in enumerate(decks):
                for target in hash_to_targets.get(value, []):
                    matched[target].append(side)
            if not matched:
                return episode_id, None, "cached_non_target"
            digest = file_hash(source)
            object_path = OUTPUT / "_objects/sha256" / f"{digest}.json"
            link(source, object_path)
            for target in matched:
                link(object_path, OUTPUT / "by_deck" / target / date / f"{episode_id}.json")
            return episode_id, {
                "episode_id": int(episode_id),
                "source_date": date,
                "create_time": metadata["create_time"],
                "qualifying_sources": metadata["qualifying_sources"],
                "agents": metadata["agents"],
                "deck_hashes": decks,
                "target_sides": dict(matched),
                "replay_sha256": digest,
                "object_path": str(object_path),
                "source_catalog": None,
                "source_cache": str(source),
            }, "cached_target"
        except Exception as exc:
            return episode_id, None, f"error:{type(exc).__name__}:{exc}"

    ids = [str(x) for x in latest["missing_catalog_episode_ids"]]
    status_counts = defaultdict(int)
    errors = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(inspect, episode_id) for episode_id in ids]
        for future in as_completed(futures):
            episode_id, row, status = future.result()
            status_counts[status.split(":", 1)[0]] += 1
            if status.startswith("error:"):
                errors.append({"episode_id": int(episode_id), "error": status})
            if row:
                existing[episode_id] = row
    merged = sorted(existing.values(), key=lambda row: row["episode_id"])
    counts = defaultdict(int)
    for row in merged:
        for target in row["target_sides"]:
            counts[target] += 1
    missing_replays = []
    for episode_id in ids:
        metadata = episode_index["episodes"][episode_id]
        if locate(episode_id, metadata["create_time"][:10]) is None:
            missing_replays.append(int(episode_id))
    latest.update(
        {
            "matching_replays": len(merged),
            "by_deck": {name: counts.get(name, 0) for name in TARGETS},
            "uncataloged_cache_scan": dict(status_counts),
            "uncataloged_scan_errors": errors,
            "missing_replay_count": len(missing_replays),
            "missing_replay_episode_ids": missing_replays,
        }
    )
    atomic_json(manifest_path, merged)
    atomic_json(latest_path, latest)
    print(json.dumps({"matching_replays": len(merged), "by_deck": latest["by_deck"], "scan": dict(status_counts), "missing_replay_count": len(missing_replays), "errors": len(errors)}))


if __name__ == "__main__":
    main()
