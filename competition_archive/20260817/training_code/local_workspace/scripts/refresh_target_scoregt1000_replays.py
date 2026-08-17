from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNS = Path("/dataT0/Free/lzhang/pocketmon-runs")
LEAGUE = RUNS / "experiment7-async-ppo-league-20260811/state/league.json"
RAW_ROOT = Path("/dataT0/Free/lzhang/pokemon_tcg_ai_battle/replays")
OUTPUT = RUNS / "experiment7-daily-replay-bc/target-scoregt1000"

CATALOGS = (
    RUNS / "experiment7-universal-20260810/prepared/catalog/replay_catalog.csv",
    RUNS / "experiment7-universal-7d-scoregt900-20260810/daily/2026-08-08/prepared/catalog/replay_catalog.csv",
    RUNS / "replay-refresh-20260812/cache/2026-08-09/prepared/catalog/replay_catalog.csv",
    RUNS / "replay-refresh-20260812/cache/2026-08-10/prepared/catalog/replay_catalog.csv",
    RUNS / "replay-refresh-20260812/cache/2026-08-11/prepared/catalog/replay_catalog.csv",
    RUNS / "experiment7-daily-replay-bc/cache/2026-08-12/prepared/catalog/replay_catalog.csv",
    RUNS / "experiment7-daily-replay-bc/cache/2026-08-13/prepared/catalog/replay_catalog.csv",
)

TARGET_CHAINS = {
    "a02_baseline": ("a02_grim_large_g9", "a02_grim_g247"),
    "a02_pokegear": ("a02_grim_large_g9_pokegear", "a02_grim_g247_pokegear"),
    "a08_maximum_belt": ("a08_maxbelt_large_g9", "a08_maxbelt"),
    "a08_rabsca": ("a08_rabsca",),
    "gold_exact_lucario": ("lucario_gold_exact",),
    "dragapult_munkidori_interference": ("dragapult_munkidori_large_g9",),
}

MANIFEST_FIELDS = (
    "episode_id",
    "source_date",
    "create_time",
    "min_score",
    "max_score",
    "sum_score",
    "deck0_sha256",
    "deck1_sha256",
    "target_decks",
    "target_sides",
    "replay_sha256",
    "object_path",
    "is_clean",
    "source_catalog",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_deck_hash(cards: list[int]) -> str:
    if len(cards) != 60 or any(not isinstance(card, int) or isinstance(card, bool) for card in cards):
        raise ValueError("deck is not exactly 60 integer card IDs")
    payload = json.dumps(tuple(sorted(cards)), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def replay_decks(payload: dict[str, Any]) -> tuple[str, str]:
    decks: list[str | None] = [None, None]
    for pair in (payload.get("steps") or [])[:12]:
        if not isinstance(pair, list):
            continue
        for player, step in enumerate(pair[:2]):
            if decks[player] is not None or not isinstance(step, dict):
                continue
            action = step.get("action")
            if isinstance(action, list) and len(action) == 60:
                decks[player] = canonical_deck_hash(action)
        if all(decks):
            break
    if not all(decks):
        raise ValueError("could not recover both 60-card decks")
    return str(decks[0]), str(decks[1])


def target_allowlist() -> tuple[dict[str, set[str]], dict[str, Any]]:
    league = load_json(LEAGUE)
    chains = league.get("chains") or {}
    allowlist: dict[str, set[str]] = {}
    receipt: dict[str, Any] = {"schemaVersion": 1, "updatedAt": now(), "targets": {}}
    for target, candidates in TARGET_CHAINS.items():
        hashes: set[str] = set()
        sources = []
        for chain_name in candidates:
            chain = chains.get(chain_name)
            if not isinstance(chain, dict):
                continue
            deck_hash = str(chain.get("deckSha256") or "")
            deck_path = Path(str(chain.get("deckPath") or ""))
            if len(deck_hash) != 64 or not deck_path.is_file():
                continue
            cards = [int(value) for value in deck_path.read_text(encoding="utf-8").split()]
            actual = canonical_deck_hash(cards)
            if actual != deck_hash:
                raise RuntimeError(f"deck hash mismatch for {chain_name}: {actual} != {deck_hash}")
            hashes.add(deck_hash)
            sources.append(
                {
                    "chain": chain_name,
                    "deckName": chain.get("deckName"),
                    "deckPath": str(deck_path),
                    "deckSha256": deck_hash,
                }
            )
        if not hashes:
            raise RuntimeError(f"failed closed: no exact deck hash for {target}")
        allowlist[target] = hashes
        receipt["targets"][target] = {"hashes": sorted(hashes), "sources": sources}
    return allowlist, receipt


def read_candidates(allowlist: dict[str, set[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hash_to_targets: dict[str, list[str]] = defaultdict(list)
    for target, hashes in allowlist.items():
        for deck_hash in hashes:
            hash_to_targets[deck_hash].append(target)

    selected: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for catalog in CATALOGS:
        if not catalog.is_file():
            raise FileNotFoundError(catalog)
        with catalog.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                episode = str(row.get("episode_id") or "")
                if not episode:
                    continue
                minimum = float(row.get("min_score") or 0)
                total = float(row.get("sum_score") or 0)
                maximum = total - minimum
                if maximum <= 1000.0:
                    continue
                deck_hashes = (str(row.get("deck0_sha256") or ""), str(row.get("deck1_sha256") or ""))
                matched: dict[str, list[int]] = defaultdict(list)
                for side, deck_hash in enumerate(deck_hashes):
                    for target in hash_to_targets.get(deck_hash, []):
                        matched[target].append(side)
                if not matched:
                    continue
                candidate = {
                    "episode_id": episode,
                    "source_date": str(row.get("source_date") or ""),
                    "create_time": str(row.get("create_time") or ""),
                    "min_score": minimum,
                    "max_score": maximum,
                    "sum_score": total,
                    "deck0_sha256": deck_hashes[0],
                    "deck1_sha256": deck_hashes[1],
                    "target_decks": ";".join(sorted(matched)),
                    "target_sides": ";".join(
                        f"{target}:{','.join(str(side) for side in sides)}"
                        for target, sides in sorted(matched.items())
                    ),
                    "replay_sha256": str(row.get("json_sha256") or ""),
                    "object_path": "",
                    "is_clean": str(row.get("is_clean") or ""),
                    "source_catalog": str(catalog),
                    "raw_path": str(row.get("raw_path") or ""),
                }
                existing = selected.get(episode)
                if existing and existing["replay_sha256"] != candidate["replay_sha256"]:
                    conflicts.append(
                        {
                            "episode_id": episode,
                            "first_sha256": existing["replay_sha256"],
                            "second_sha256": candidate["replay_sha256"],
                            "first_catalog": existing["source_catalog"],
                            "second_catalog": candidate["source_catalog"],
                        }
                    )
                    continue
                selected[episode] = candidate
    return sorted(selected.values(), key=lambda row: (row["source_date"], int(row["episode_id"]))), conflicts


def locate_existing(row: dict[str, Any]) -> Path | None:
    candidates = [
        Path(row["raw_path"]),
        RAW_ROOT / row["source_date"] / f"{row['episode_id']}.json",
    ]
    for staging in sorted((RAW_ROOT / "_staging").glob("*"), reverse=True):
        candidates.append(staging / row["source_date"] / f"{row['episode_id']}.json")
    return next((path for path in candidates if path.is_file()), None)


def download_one(row: dict[str, Any], retries: int) -> tuple[str, Path | None, str | None]:
    existing = locate_existing(row)
    if existing:
        return row["episode_id"], existing, None
    slug = f"kaggle/pokemon-tcg-ai-battle-episodes-{row['source_date']}"
    filename = f"{row['episode_id']}.json"
    error = "not attempted"
    download_temp_root = OUTPUT / "_tmp"
    download_temp_root.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with tempfile.TemporaryDirectory(
                prefix=f"ptcg-{row['episode_id']}-", dir=download_temp_root
            ) as temp_name:
                temporary = Path(temp_name)
                completed = subprocess.run(
                    [sys.executable, "-m", "kaggle", "datasets", "download", slug, "-f", filename, "-p", str(temporary)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                if completed.returncode:
                    raise RuntimeError((completed.stderr or completed.stdout).strip())
                source = temporary / filename
                archive = temporary / f"{filename}.zip"
                if archive.is_file():
                    with zipfile.ZipFile(archive) as handle:
                        handle.extract(filename, temporary)
                if not source.is_file():
                    raise FileNotFoundError(filename)
                destination = RAW_ROOT / row["source_date"] / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                staged = destination.with_name(f".{filename}.{os.getpid()}.tmp")
                shutil.copy2(source, staged)
                os.replace(staged, destination)
                return row["episode_id"], destination, None
        except Exception as exc:
            error = f"attempt {attempt}: {exc}"
            time.sleep(min(2**attempt, 8))
    return row["episode_id"], None, error


def link_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    try:
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_and_materialize(row: dict[str, Any], source: Path, allowlist: dict[str, set[str]]) -> None:
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if row["replay_sha256"] and digest != row["replay_sha256"]:
        raise RuntimeError(f"content hash mismatch for episode {row['episode_id']}")
    payload = json.loads(raw)
    episode = int((payload.get("info") or {}).get("EpisodeId"))
    if episode != int(row["episode_id"]):
        raise RuntimeError(f"episode ID mismatch: {episode} != {row['episode_id']}")
    actual_decks = replay_decks(payload)
    if actual_decks != (row["deck0_sha256"], row["deck1_sha256"]):
        raise RuntimeError(f"deck hash mismatch for episode {row['episode_id']}")
    object_path = OUTPUT / "_objects/sha256" / f"{digest}.json"
    link_atomic(source, object_path)
    row["replay_sha256"] = digest
    row["object_path"] = str(object_path)
    for target in row["target_decks"].split(";"):
        if not target:
            continue
        if not ({actual_decks[0], actual_decks[1]} & allowlist[target]):
            raise RuntimeError(f"target mapping mismatch for {target} episode {episode}")
        link_atomic(
            object_path,
            OUTPUT / "by_deck" / target / row["source_date"] / f"{row['episode_id']}.json",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh score>1000 target-deck replay cache")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.workers <= 0 or args.retries <= 0:
        raise ValueError("workers and retries must be positive")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock_handle = (OUTPUT / "refresh.lock").open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps({"status": "busy", "lock": str(OUTPUT / "refresh.lock")}))
        return 0

    allowlist, deck_receipt = target_allowlist()
    rows, catalog_conflicts = read_candidates(allowlist)
    existing = {row["episode_id"]: locate_existing(row) for row in rows}
    missing_rows = [row for row in rows if existing[row["episode_id"]] is None]
    counts = defaultdict(int)
    for row in rows:
        for target in row["target_decks"].split(";"):
            counts[target] += 1
    plan = {
        "checkedAt": now(),
        "scoreRule": "max_score = sum_score - min_score > 1000",
        "catalogs": [str(path) for path in CATALOGS],
        "eligibleUnique": len(rows),
        "availableLocally": len(rows) - len(missing_rows),
        "missingBeforeDownload": len(missing_rows),
        "catalogConflicts": catalog_conflicts,
        "byDeck": dict(sorted(counts.items())),
        "targetDecks": deck_receipt["targets"],
    }
    print(json.dumps(plan, ensure_ascii=False), flush=True)
    if args.plan_only:
        return 0

    atomic_json(OUTPUT / "target_decks.json", deck_receipt)
    atomic_json(OUTPUT / "monitoring/plan.json", plan)

    materialized: list[dict[str, Any]] = []
    validation_errors: list[dict[str, str]] = []
    available_rows = [row for row in rows if existing[row["episode_id"]] is not None]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                validate_and_materialize, row, existing[row["episode_id"]], allowlist
            ): row
            for row in available_rows
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                future.result()
                materialized.append(row)
            except Exception as exc:
                validation_errors.append({"episode_id": row["episode_id"], "error": str(exc)})
            if index % 250 == 0 or index == len(available_rows):
                progress = {
                    "status": "running",
                    "checkedAt": now(),
                    "existingProgress": f"{index}/{len(available_rows)}",
                    "eligible": len(rows),
                    "materializedUnique": len(materialized),
                    "validationErrors": len(validation_errors),
                }
                atomic_json(OUTPUT / "monitoring/latest.json", progress)
                print(json.dumps(progress), flush=True)

    download_errors: list[dict[str, str]] = []
    if missing_rows:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(download_one, row, args.retries): row for row in missing_rows}
            for index, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                episode, path, error = future.result()
                existing[episode] = path
                if error or path is None:
                    download_errors.append({"episode_id": episode, "error": error or "download returned no path"})
                else:
                    try:
                        validate_and_materialize(row, path, allowlist)
                        materialized.append(row)
                    except Exception as exc:
                        validation_errors.append({"episode_id": episode, "error": str(exc)})
                if index % 25 == 0 or index == len(futures):
                    progress = {
                        "status": "running",
                        "checkedAt": now(),
                        "downloadProgress": f"{index}/{len(futures)}",
                        "eligible": len(rows),
                        "materializedUnique": len(materialized),
                        "downloadErrors": len(download_errors),
                        "validationErrors": len(validation_errors),
                    }
                    atomic_json(OUTPUT / "monitoring/latest.json", progress)
                    print(json.dumps(progress), flush=True)

    export_rows = [{key: row.get(key, "") for key in MANIFEST_FIELDS} for row in materialized]
    atomic_csv(OUTPUT / "monitoring/global_manifest.csv", export_rows)
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in export_rows:
        for target in str(row["target_decks"]).split(";"):
            if target:
                by_target[target].append(row)
    for target in TARGET_CHAINS:
        atomic_csv(OUTPUT / "by_deck" / target / "index.csv", by_target.get(target, []))

    dates = sorted({row["source_date"] for row in materialized})
    latest = {
        "schemaVersion": 1,
        "status": "complete" if not download_errors and not validation_errors else "partial",
        "checkedAt": now(),
        "scoreRule": plan["scoreRule"],
        "datesChecked": dates,
        "eligible": len(rows),
        "newDownloaded": len(missing_rows) - len(download_errors),
        "alreadyPresent": len(rows) - len(missing_rows),
        "materializedUnique": len(materialized),
        "missing": download_errors,
        "validationErrors": validation_errors,
        "conflicts": catalog_conflicts,
        "byDeck": {target: len(by_target.get(target, [])) for target in TARGET_CHAINS},
        "bytes": sum((OUTPUT / "_objects/sha256" / f"{row['replay_sha256']}.json").stat().st_size for row in materialized),
    }
    atomic_json(OUTPUT / "monitoring/latest.json", latest)
    atomic_json(OUTPUT / "monitoring/watermark.json", {"checkedAt": latest["checkedAt"], "dates": dates})
    print(json.dumps(latest, ensure_ascii=False), flush=True)
    return 0 if latest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
