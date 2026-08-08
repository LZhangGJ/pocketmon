from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CURRENT_DATE = "2026-08-07"
PREVIOUS_DATE = "2026-08-06"
DATES = (PREVIOUS_DATE, CURRENT_DATE)
ELITE_FRACTION = 0.10
ARCHETYPE_SIMILARITY = 0.55
OUTPUT_DIR = ROOT / "analysis" / "outputs" / "top_ladder_2026_08_07"
REPLAY_ROOT = ROOT / "data" / "raw" / "replays"
CARD_ROOT = ROOT / "data" / "raw" / "cards"
EN_CARDS = CARD_ROOT / "EN%20Card%20Data.csv"
JP_CARDS = CARD_ROOT / "JP%20Card%20Data.csv"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _deck_signature(deck: list[int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(Counter(deck).items()))


def _signature_id(signature: tuple[tuple[int, int], ...], prefix: str) -> str:
    payload = json.dumps(signature, separators=(",", ":"), ensure_ascii=True)
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _parse_replay(task: dict[str, Any]) -> dict[str, Any]:
    path = Path(task["path"])
    result: dict[str, Any] = {
        "episode_id": int(task["episode_id"]),
        "date": task["date"],
        "path": str(path),
        "file_present": path.exists(),
        "parse_ok": False,
        "deck_complete": False,
        "valid_outcome": False,
        "status_done": False,
        "first_player": None,
        "teams": [],
        "rewards": [],
        "statuses": [],
        "decks": [None, None],
        "error": None,
    }
    if not path.exists():
        result["error"] = "missing_file"
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            replay = json.load(handle)
    except Exception as exc:  # recorded for quality reporting
        result["error"] = f"{type(exc).__name__}: {exc}"[:240]
        return result

    result["parse_ok"] = True
    info = replay.get("info") or {}
    teams = info.get("TeamNames") or [
        item.get("Name", "") for item in (info.get("Agents") or []) if isinstance(item, dict)
    ]
    result["teams"] = list(teams[:2])
    rewards = replay.get("rewards") or []
    statuses = replay.get("statuses") or []
    result["rewards"] = list(rewards[:2])
    result["statuses"] = list(statuses[:2])
    result["status_done"] = len(statuses) >= 2 and all(value == "DONE" for value in statuses[:2])

    decks: list[list[int] | None] = [None, None]
    first_player = None
    for pair in (replay.get("steps") or [])[:8]:
        if not isinstance(pair, list):
            continue
        for agent_index, agent_step in enumerate(pair[:2]):
            if not isinstance(agent_step, dict):
                continue
            action = agent_step.get("action")
            if (
                decks[agent_index] is None
                and isinstance(action, list)
                and len(action) == 60
                and all(isinstance(card_id, int) for card_id in action)
            ):
                decks[agent_index] = action
            if first_player is None:
                observation = agent_step.get("observation") or {}
                current = observation.get("current") if isinstance(observation, dict) else None
                if isinstance(current, dict) and current.get("firstPlayer") in (0, 1):
                    first_player = int(current["firstPlayer"])
        if all(deck is not None for deck in decks) and first_player is not None:
            break

    result["first_player"] = first_player
    result["decks"] = decks
    result["deck_complete"] = all(
        isinstance(deck, list) and len(deck) == 60 for deck in decks
    )
    result["valid_outcome"] = (
        len(rewards) >= 2
        and all(_is_number(value) for value in rewards[:2])
        and abs(float(rewards[0]) + float(rewards[1])) < 1e-9
        and all(float(value) in (-1.0, 0.0, 1.0) for value in rewards[:2])
    )
    return result


def _load_cards() -> pd.DataFrame:
    if not EN_CARDS.exists() or not JP_CARDS.exists():
        raise FileNotFoundError(
            "Official card CSVs are missing. Download EN Card Data.csv and JP Card Data.csv first."
        )
    english = pd.read_csv(EN_CARDS, encoding="utf-8-sig").rename(
        columns={"Card ID": "card_id", "Card Name": "card_name_en"}
    )
    english = english.drop_duplicates("card_id", keep="first").copy()
    japanese = pd.read_csv(JP_CARDS, encoding="utf-8-sig").rename(
        columns={"カード ID": "card_id", "カード名": "card_name_jp"}
    )
    japanese = japanese.drop_duplicates("card_id", keep="first").copy()
    cards = english.merge(japanese[["card_id", "card_name_jp"]], on="card_id", how="left")
    cards["card_id"] = cards["card_id"].astype(int)
    subtype = cards["Stage (Pokémon)/Type (Energy and Trainer)"].fillna("").astype(str)
    cards["card_subtype"] = subtype
    cards["is_pokemon"] = subtype.str.endswith("Pokémon")
    cards["card_group"] = np.select(
        [
            cards["is_pokemon"],
            subtype.str.contains("Energy", case=False, na=False),
        ],
        ["Pokémon", "Energy"],
        default="Trainer",
    )
    return cards


def _load_date(date: str, workers: int, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_path = REPLAY_ROOT / date / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    cache_dir = OUTPUT_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    games_cache = cache_dir / f"games_{date}.pkl"
    decks_cache = cache_dir / f"decks_{date}.pkl"
    meta_cache = cache_dir / f"meta_{date}.json"
    source_signature = {
        "manifest_mtime_ns": manifest_path.stat().st_mtime_ns,
        "json_file_count": len(list((REPLAY_ROOT / date).glob("*.json"))),
    }
    if not force and games_cache.exists() and decks_cache.exists() and meta_cache.exists():
        cached_meta = json.loads(meta_cache.read_text(encoding="utf-8"))
        if cached_meta.get("source_signature") == source_signature:
            return (
                pd.read_pickle(games_cache),
                pd.read_pickle(decks_cache),
                cached_meta["quality"],
            )
    manifest = pd.read_csv(manifest_path)
    manifest["episode_id"] = manifest["episode_id"].astype(int)
    manifest["create_time"] = pd.to_datetime(manifest["create_time"], errors="coerce")
    manifest["date"] = date
    tasks = [
        {
            "episode_id": int(row.episode_id),
            "date": date,
            "path": str(REPLAY_ROOT / date / f"{int(row.episode_id)}.json"),
        }
        for row in manifest.itertuples(index=False)
    ]

    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        parsed = pool.map(_parse_replay, tasks, chunksize=max(1, len(tasks) // (workers * 8)))

    game_rows: list[dict[str, Any]] = []
    deck_rows: list[dict[str, Any]] = []
    for record in parsed:
        game = {key: value for key, value in record.items() if key not in {"decks", "teams", "rewards", "statuses"}}
        game["team_0"] = record["teams"][0] if len(record["teams"]) > 0 else ""
        game["team_1"] = record["teams"][1] if len(record["teams"]) > 1 else ""
        game["reward_0"] = record["rewards"][0] if len(record["rewards"]) > 0 else None
        game["reward_1"] = record["rewards"][1] if len(record["rewards"]) > 1 else None
        game["status_0"] = record["statuses"][0] if len(record["statuses"]) > 0 else ""
        game["status_1"] = record["statuses"][1] if len(record["statuses"]) > 1 else ""
        game["valid_game"] = bool(record["deck_complete"] and record["valid_outcome"])
        game_rows.append(game)

        if record["deck_complete"]:
            for player_index, deck in enumerate(record["decks"]):
                assert isinstance(deck, list)
                signature = _deck_signature(deck)
                reward = record["rewards"][player_index] if len(record["rewards"]) > player_index else None
                deck_rows.append(
                    {
                        "episode_id": int(record["episode_id"]),
                        "date": date,
                        "player_index": player_index,
                        "team_name": record["teams"][player_index] if len(record["teams"]) > player_index else "",
                        "reward": reward,
                        "win_value": (1.0 if reward > 0 else 0.5 if reward == 0 else 0.0) if _is_number(reward) else np.nan,
                        "is_first_player": record["first_player"] == player_index if record["first_player"] in (0, 1) else None,
                        "deck_list": deck,
                        "exact_signature": signature,
                        "exact_deck_id": _signature_id(signature, "deck"),
                    }
                )

    games = manifest.merge(pd.DataFrame(game_rows), on=["episode_id", "date"], how="left", validate="one_to_one")
    decks = pd.DataFrame(deck_rows).merge(
        manifest[["episode_id", "date", "create_time", "avg_score", "min_score", "sum_score"]],
        on=["episode_id", "date"],
        how="left",
        validate="many_to_one",
    )
    valid_ids = set(games.loc[games["valid_game"].fillna(False), "episode_id"])
    decks["valid_game"] = decks["episode_id"].isin(valid_ids)

    quality = {
        "date": date,
        "manifest_games": int(len(manifest)),
        "manifest_duplicate_episode_ids": int(manifest["episode_id"].duplicated().sum()),
        "json_files_present": int(len(list((REPLAY_ROOT / date).glob("*.json")))),
        "files_present_for_manifest": int(games["file_present"].fillna(False).sum()),
        "parsed_games": int(games["parse_ok"].fillna(False).sum()),
        "deck_complete_games": int(games["deck_complete"].fillna(False).sum()),
        "valid_outcome_games": int(games["valid_outcome"].fillna(False).sum()),
        "valid_games": int(games["valid_game"].fillna(False).sum()),
        "status_done_games": int(games["status_done"].fillna(False).sum()),
    }
    quality["valid_game_coverage"] = quality["valid_games"] / quality["manifest_games"] if quality["manifest_games"] else 0.0
    games.to_pickle(games_cache)
    decks.to_pickle(decks_cache)
    meta_cache.write_text(
        json.dumps(
            {"source_signature": source_signature, "quality": quality},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return games, decks, quality


def _weighted_jaccard(
    left: tuple[tuple[int, int], ...],
    right: tuple[tuple[int, int], ...],
    idf: dict[int, float],
) -> float:
    left_map = dict(left)
    right_map = dict(right)
    card_ids = set(left_map) | set(right_map)
    numerator = sum(min(left_map.get(card_id, 0), right_map.get(card_id, 0)) * idf.get(card_id, 0.15) for card_id in card_ids)
    denominator = sum(max(left_map.get(card_id, 0), right_map.get(card_id, 0)) * idf.get(card_id, 0.15) for card_id in card_ids)
    return numerator / denominator if denominator else 0.0


def _cluster_archetypes(
    current_elite: pd.DataFrame,
    previous_elite: pd.DataFrame,
    cards: pd.DataFrame,
) -> tuple[dict[tuple[tuple[int, int], ...], str], pd.DataFrame, dict[int, float]]:
    combined = pd.concat([current_elite, previous_elite], ignore_index=True)
    profiles = list(dict.fromkeys(combined["pokemon_signature"].tolist()))
    total_appearances = len(combined)
    inclusion = Counter(
        card_id
        for signature in combined["pokemon_signature"]
        for card_id, _ in signature
    )
    idf = {
        card_id: math.log((total_appearances + 1) / (count + 1)) + 0.15
        for card_id, count in inclusion.items()
    }

    parents = list(range(len(profiles)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(profiles)):
        for right in range(left):
            if _weighted_jaccard(profiles[left], profiles[right], idf) >= ARCHETYPE_SIMILARITY:
                union(left, right)

    components: dict[int, list[tuple[tuple[int, int], ...]]] = defaultdict(list)
    for index, profile in enumerate(profiles):
        components[find(index)].append(profile)

    card_lookup = cards.set_index("card_id").to_dict("index")
    current_counts = Counter(current_elite["pokemon_signature"])
    total_counts = Counter(combined["pokemon_signature"])

    component_rows: list[dict[str, Any]] = []
    for component_profiles in components.values():
        representative = max(
            component_profiles,
            key=lambda profile: (current_counts[profile], total_counts[profile], profile),
        )
        current_n = sum(current_counts[profile] for profile in component_profiles)
        previous_n = sum(
            int(previous_elite["pokemon_signature"].map(lambda value: value == profile).sum())
            for profile in component_profiles
        )

        def label_sort(item: tuple[int, int]) -> tuple[float, float, float, int]:
            card_id, copies = item
            meta = card_lookup.get(card_id, {})
            name = str(meta.get("card_name_en", ""))
            subtype = str(meta.get("card_subtype", ""))
            ex_bonus = 2.0 if " ex" in name.lower() else 0.0
            stage_bonus = 1.5 if subtype.startswith("Stage 2") else 1.0 if subtype.startswith("Stage 1") else 0.0
            distinctiveness = copies * idf.get(card_id, 0.15)
            return (ex_bonus, stage_bonus, distinctiveness, copies)

        anchor_cards = sorted(representative, key=label_sort, reverse=True)[:3]
        label = " / ".join(
            str(card_lookup.get(card_id, {}).get("card_name_en") or f"Card {card_id}")
            for card_id, _ in anchor_cards
        ) or "Unclassified Pokémon core"
        component_rows.append(
            {
                "profiles": component_profiles,
                "representative": representative,
                "label": label,
                "current_appearances": current_n,
                "previous_appearances": previous_n,
            }
        )

    component_rows.sort(
        key=lambda row: (row["current_appearances"], row["previous_appearances"], row["label"]),
        reverse=True,
    )
    seen_labels: Counter[str] = Counter()
    mapping: dict[tuple[tuple[int, int], ...], str] = {}
    catalog_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(component_rows, start=1):
        archetype_id = f"A{rank:02d}"
        seen_labels[row["label"]] += 1
        label = row["label"]
        if seen_labels[label] > 1:
            label = f"{label} ({seen_labels[label]})"
        for profile in row["profiles"]:
            mapping[profile] = archetype_id
        catalog_rows.append(
            {
                "archetype_id": archetype_id,
                "archetype_label": label,
                "current_appearances": int(row["current_appearances"]),
                "previous_appearances": int(row["previous_appearances"]),
                "pokemon_profile_variants": len(row["profiles"]),
                "representative_signature": row["representative"],
            }
        )
    return mapping, pd.DataFrame(catalog_rows), idf


def _wilson(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (np.nan, np.nan)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    spread = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    return (max(0.0, center - spread), min(1.0, center + spread))


def _add_opponent(decks: pd.DataFrame) -> pd.DataFrame:
    opponent = decks[
        ["episode_id", "player_index", "archetype_id", "archetype_label", "team_name", "exact_deck_id"]
    ].copy()
    opponent["player_index"] = 1 - opponent["player_index"]
    opponent = opponent.rename(
        columns={
            "archetype_id": "opponent_archetype_id",
            "archetype_label": "opponent_archetype_label",
            "team_name": "opponent_team_name",
            "exact_deck_id": "opponent_exact_deck_id",
        }
    )
    return decks.merge(opponent, on=["episode_id", "player_index"], how="left", validate="one_to_one")


def _archetype_summary(elite: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = _add_opponent(elite)
    rows: list[dict[str, Any]] = []
    total_appearances = len(paired)
    for (archetype_id, label), group in paired.groupby(["archetype_id", "archetype_label"], sort=False):
        nonmirror = group[group["opponent_archetype_id"] != archetype_id]
        wins = float(group["win_value"].sum())
        nonmirror_wins = float(nonmirror["win_value"].sum())
        lower, upper = _wilson(nonmirror_wins, len(nonmirror))
        rows.append(
            {
                "archetype_id": archetype_id,
                "archetype_label": label,
                "appearances": int(len(group)),
                "usage_share": len(group) / total_appearances if total_appearances else 0.0,
                "all_win_rate": wins / len(group) if len(group) else np.nan,
                "nonmirror_appearances": int(len(nonmirror)),
                "nonmirror_win_rate": nonmirror_wins / len(nonmirror) if len(nonmirror) else np.nan,
                "nonmirror_ci_low": lower,
                "nonmirror_ci_high": upper,
                "unique_teams": int(group["team_name"].nunique()),
                "exact_deck_variants": int(group["exact_deck_id"].nunique()),
                "avg_game_min_score": float(group["min_score"].mean()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["appearances", "archetype_label"], ascending=[False, True]).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))

    matchup = (
        paired.groupby(
            ["archetype_id", "archetype_label", "opponent_archetype_id", "opponent_archetype_label"],
            dropna=False,
        )
        .agg(matchup_appearances=("episode_id", "size"), wins=("win_value", "sum"))
        .reset_index()
    )
    matchup["matchup_win_rate"] = matchup["wins"] / matchup["matchup_appearances"]
    return summary, matchup


def _card_usage(
    elite: pd.DataFrame,
    field: pd.DataFrame,
    cards: pd.DataFrame,
) -> pd.DataFrame:
    elite_inclusion: Counter[int] = Counter()
    elite_copies: Counter[int] = Counter()
    for deck in elite["deck_list"]:
        counts = Counter(deck)
        elite_inclusion.update(counts.keys())
        elite_copies.update(counts)

    field_inclusion: Counter[int] = Counter()
    for deck in field["deck_list"]:
        field_inclusion.update(set(deck))

    card_lookup = cards.set_index("card_id").to_dict("index")
    rows = []
    elite_total = len(elite)
    field_total = len(field)
    for card_id, inclusion_count in elite_inclusion.items():
        meta = card_lookup.get(card_id, {})
        elite_rate = inclusion_count / elite_total if elite_total else 0.0
        field_rate = field_inclusion[card_id] / field_total if field_total else 0.0
        rows.append(
            {
                "card_id": card_id,
                "card_name_en": meta.get("card_name_en", f"Card {card_id}"),
                "card_name_jp": meta.get("card_name_jp", ""),
                "card_group": meta.get("card_group", "Unknown"),
                "card_subtype": meta.get("card_subtype", ""),
                "elite_inclusion_count": inclusion_count,
                "elite_inclusion_rate": elite_rate,
                "avg_copies_per_elite_deck": elite_copies[card_id] / elite_total if elite_total else 0.0,
                "avg_copies_when_included": elite_copies[card_id] / inclusion_count,
                "field_inclusion_rate": field_rate,
                "elite_vs_field_lift": (elite_rate + 1 / max(elite_total, 1)) / (field_rate + 1 / max(field_total, 1)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["elite_inclusion_rate", "avg_copies_per_elite_deck"], ascending=False
    ).reset_index(drop=True)


def _representative_decklists(
    elite: pd.DataFrame,
    summary: pd.DataFrame,
    cards: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    card_lookup = cards.set_index("card_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    for summary_row in summary.head(top_n).itertuples(index=False):
        group = elite[elite["archetype_id"] == summary_row.archetype_id]
        representative_deck_id = group["exact_deck_id"].value_counts().index[0]
        representative = group[group["exact_deck_id"] == representative_deck_id].iloc[0]
        for card_id, count in sorted(Counter(representative["deck_list"]).items()):
            meta = card_lookup.get(card_id, {})
            rows.append(
                {
                    "archetype_rank": int(summary_row.rank),
                    "archetype_id": summary_row.archetype_id,
                    "archetype_label": summary_row.archetype_label,
                    "representative_exact_deck_id": representative_deck_id,
                    "card_id": card_id,
                    "card_name_en": meta.get("card_name_en", f"Card {card_id}"),
                    "card_name_jp": meta.get("card_name_jp", ""),
                    "card_group": meta.get("card_group", "Unknown"),
                    "card_subtype": meta.get("card_subtype", ""),
                    "count": count,
                }
            )
    result = pd.DataFrame(rows)
    group_order = pd.CategoricalDtype(["Pokémon", "Trainer", "Energy", "Unknown"], ordered=True)
    result["card_group"] = result["card_group"].astype(group_order)
    return result.sort_values(["archetype_rank", "card_group", "card_id"]).reset_index(drop=True)


def _team_summary(elite: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team_name, group in elite.groupby("team_name"):
        primary = group["archetype_label"].value_counts()
        rows.append(
            {
                "team_name": team_name,
                "appearances": int(len(group)),
                "win_rate": float(group["win_value"].mean()),
                "primary_archetype": primary.index[0],
                "primary_archetype_share": float(primary.iloc[0] / len(group)),
                "archetypes_used": int(group["archetype_id"].nunique()),
                "exact_decks_used": int(group["exact_deck_id"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["appearances", "win_rate"], ascending=False).reset_index(drop=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def run_analysis(force: bool = False, workers: int | None = None) -> dict[str, Any]:
    workers = workers or min(16, max(2, os.cpu_count() or 2))
    cards = _load_cards()
    pokemon_ids = set(cards.loc[cards["is_pokemon"], "card_id"].astype(int))

    games_by_date: dict[str, pd.DataFrame] = {}
    decks_by_date: dict[str, pd.DataFrame] = {}
    quality_rows: list[dict[str, Any]] = []
    for date in DATES:
        games, decks, quality = _load_date(date, workers, force=force)
        decks["pokemon_signature"] = decks["deck_list"].map(
            lambda deck: tuple(sorted(Counter(card_id for card_id in deck if card_id in pokemon_ids).items()))
        )
        games_by_date[date] = games
        decks_by_date[date] = decks
        quality_rows.append(quality)

    elite_ids: dict[str, set[int]] = {}
    thresholds: dict[str, float] = {}
    target_counts: dict[str, int] = {}
    for date in DATES:
        manifest_order = games_by_date[date].sort_values(
            ["min_score", "avg_score", "episode_id"], ascending=[False, False, False]
        )
        target_count = int(math.ceil(len(manifest_order) * ELITE_FRACTION))
        selected = manifest_order.head(target_count)
        elite_ids[date] = set(selected["episode_id"].astype(int))
        thresholds[date] = float(selected["min_score"].min())
        target_counts[date] = target_count

    current_field = decks_by_date[CURRENT_DATE][decks_by_date[CURRENT_DATE]["valid_game"]].copy()
    current_elite = current_field[current_field["episode_id"].isin(elite_ids[CURRENT_DATE])].copy()
    previous_field = decks_by_date[PREVIOUS_DATE][decks_by_date[PREVIOUS_DATE]["valid_game"]].copy()
    previous_elite = previous_field[previous_field["episode_id"].isin(elite_ids[PREVIOUS_DATE])].copy()

    profile_to_archetype, catalog, _ = _cluster_archetypes(current_elite, previous_elite, cards)
    label_map = catalog.set_index("archetype_id")["archetype_label"].to_dict()
    for frame in (current_elite, previous_elite):
        frame["archetype_id"] = frame["pokemon_signature"].map(profile_to_archetype)
        frame["archetype_label"] = frame["archetype_id"].map(label_map)

    archetype_summary, matchups = _archetype_summary(current_elite)
    current_total = len(current_elite)
    previous_total = len(previous_elite)
    current_counts = current_elite["archetype_id"].value_counts()
    previous_counts = previous_elite["archetype_id"].value_counts()
    day_comparison = catalog[["archetype_id", "archetype_label"]].copy()
    day_comparison["current_appearances"] = day_comparison["archetype_id"].map(current_counts).fillna(0).astype(int)
    day_comparison["previous_appearances"] = day_comparison["archetype_id"].map(previous_counts).fillna(0).astype(int)
    day_comparison["current_usage_share"] = day_comparison["current_appearances"] / current_total if current_total else 0.0
    day_comparison["previous_usage_share"] = day_comparison["previous_appearances"] / previous_total if previous_total else 0.0
    day_comparison["usage_share_delta_pp"] = 100 * (
        day_comparison["current_usage_share"] - day_comparison["previous_usage_share"]
    )
    day_comparison = day_comparison.sort_values("current_appearances", ascending=False).reset_index(drop=True)

    card_usage = _card_usage(current_elite, current_field, cards)
    representative_decklists = _representative_decklists(current_elite, archetype_summary, cards)
    team_summary = _team_summary(current_elite)

    top_ids = set(archetype_summary.head(8)["archetype_id"])
    matchup_top = matchups[
        matchups["archetype_id"].isin(top_ids)
        & matchups["opponent_archetype_id"].isin(top_ids)
    ].copy()
    matchup_matrix = matchup_top.pivot(
        index="archetype_label",
        columns="opponent_archetype_label",
        values="matchup_win_rate",
    )
    matchup_n_matrix = matchup_top.pivot(
        index="archetype_label",
        columns="opponent_archetype_label",
        values="matchup_appearances",
    )

    current_elite_games = games_by_date[CURRENT_DATE][
        games_by_date[CURRENT_DATE]["episode_id"].isin(elite_ids[CURRENT_DATE])
        & games_by_date[CURRENT_DATE]["valid_game"].fillna(False)
    ]
    first_player_rows = current_elite[current_elite["is_first_player"] == True]  # noqa: E712
    first_player_win_rate = float(first_player_rows["win_value"].mean()) if len(first_player_rows) else np.nan

    top_row = archetype_summary.iloc[0]
    top_three = archetype_summary.head(3)[
        ["archetype_label", "appearances", "usage_share", "nonmirror_win_rate", "nonmirror_appearances"]
    ].to_dict("records")
    previous_lookup = day_comparison.set_index("archetype_id")
    top_previous_share = float(previous_lookup.loc[top_row.archetype_id, "previous_usage_share"])
    summary = {
        "as_of_date": CURRENT_DATE,
        "previous_date": PREVIOUS_DATE,
        "source_latest_create_time": games_by_date[CURRENT_DATE]["create_time"].max(),
        "timezone_note": "Kaggle manifest create_time has no timezone offset; timestamps are reported as source time.",
        "elite_definition": f"Top {ELITE_FRACTION:.0%} of daily matches ranked by min_score, so both players clear the cutoff.",
        "elite_min_score_cutoff": thresholds[CURRENT_DATE],
        "previous_elite_min_score_cutoff": thresholds[PREVIOUS_DATE],
        "elite_target_games": target_counts[CURRENT_DATE],
        "elite_valid_games": int(len(current_elite_games)),
        "elite_deck_appearances": int(len(current_elite)),
        "elite_unique_teams": int(current_elite["team_name"].nunique()),
        "current_manifest_games": int(len(games_by_date[CURRENT_DATE])),
        "current_valid_game_coverage": float(
            games_by_date[CURRENT_DATE]["valid_game"].fillna(False).mean()
        ),
        "first_player_win_rate": first_player_win_rate,
        "top_archetype": top_row.archetype_label,
        "top_archetype_usage_share": float(top_row.usage_share),
        "top_archetype_nonmirror_win_rate": float(top_row.nonmirror_win_rate),
        "top_archetype_nonmirror_appearances": int(top_row.nonmirror_appearances),
        "top_archetype_previous_usage_share": top_previous_share,
        "top_archetype_usage_delta_pp": 100 * (float(top_row.usage_share) - top_previous_share),
        "top_three_archetypes": top_three,
    }

    quality = pd.DataFrame(quality_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    games_export = pd.concat(games_by_date.values(), ignore_index=True)
    decks_export = pd.concat([current_elite.assign(cohort="current_elite"), previous_elite.assign(cohort="previous_elite")], ignore_index=True)
    decks_export["deck_list_json"] = decks_export["deck_list"].map(lambda deck: json.dumps(deck, separators=(",", ":")))
    decks_export["pokemon_signature_json"] = decks_export["pokemon_signature"].map(lambda value: json.dumps(value, separators=(",", ":")))
    decks_export = decks_export.drop(columns=["deck_list", "exact_signature", "pokemon_signature"])

    games_export.to_csv(OUTPUT_DIR / "games.csv", index=False)
    decks_export.to_csv(OUTPUT_DIR / "elite_deck_appearances.csv", index=False)
    archetype_summary.to_csv(OUTPUT_DIR / "archetype_summary.csv", index=False)
    day_comparison.to_csv(OUTPUT_DIR / "day_comparison.csv", index=False)
    card_usage.to_csv(OUTPUT_DIR / "card_usage.csv", index=False)
    matchups.to_csv(OUTPUT_DIR / "matchups_long.csv", index=False)
    matchup_matrix.to_csv(OUTPUT_DIR / "matchup_win_rate_matrix.csv")
    matchup_n_matrix.to_csv(OUTPUT_DIR / "matchup_sample_size_matrix.csv")
    representative_decklists.to_csv(OUTPUT_DIR / "representative_decklists.csv", index=False)
    team_summary.to_csv(OUTPUT_DIR / "team_summary.csv", index=False)
    quality.to_csv(OUTPUT_DIR / "data_quality.csv", index=False)
    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, ensure_ascii=False, indent=2)

    notes = f"""# Source notes: top-ladder deck analysis

- Current source: `{REPLAY_ROOT / CURRENT_DATE}` plus its `manifest.csv`.
- Comparison source: `{REPLAY_ROOT / PREVIOUS_DATE}` plus its `manifest.csv`.
- Card mapping: official competition `EN Card Data.csv` and `JP Card Data.csv`.
- Elite cohort: top {ELITE_FRACTION:.0%} of matches by `min_score`; current cutoff {thresholds[CURRENT_DATE]:.3f}.
- Analysis grain: one deck appearance per player per valid two-player replay.
- Archetypes: exact Pokémon-card multisets connected at IDF-weighted Jaccard >= {ARCHETYPE_SIMILARITY:.2f}; Trainer/Energy differences remain exact-deck variants.
- Win rate: reward > 0 is a win, reward = 0 is a half-win, reward < 0 is a loss. Non-mirror Wilson intervals are descriptive; repeated games by the same team are not independent.
- Score-to-player mapping is unavailable in the daily manifest; `min_score` is used so both players meet the cohort cutoff.
- Source timestamps have no timezone offset.

## Chart map

1. Meta share: ranked horizontal bar; archetype vs usage share; current elite cohort.
2. Day-over-day share: grouped bar; current vs previous elite usage share; top current archetypes.
3. Performance vs presence: scatter; usage share vs non-mirror win rate with sample size retained in the source table.
4. Card inclusion: ranked horizontal bar; top card inclusion rates; current elite cohort.
"""
    (OUTPUT_DIR / "source_notes.md").write_text(notes, encoding="utf-8")

    return {
        "summary": summary,
        "quality": quality,
        "archetype_summary": archetype_summary,
        "day_comparison": day_comparison,
        "card_usage": card_usage,
        "matchups": matchups,
        "matchup_matrix": matchup_matrix,
        "matchup_n_matrix": matchup_n_matrix,
        "representative_decklists": representative_decklists,
        "team_summary": team_summary,
        "output_dir": OUTPUT_DIR,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze top-ladder Pokémon TCG replay decks")
    parser.add_argument("--workers", type=int, default=min(16, max(2, os.cpu_count() or 2)))
    args = parser.parse_args()
    result = run_analysis(workers=args.workers)
    print(json.dumps(_json_safe(result["summary"]), ensure_ascii=False, indent=2))
    print(f"output_dir={result['output_dir']}")


if __name__ == "__main__":
    main()
