#!/usr/bin/env python3
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import sys


CHAINS = [
    "a02_grim_large_g9_pokegear",
    "a08_maxbelt_large_g9",
    "dragapult_munkidori_large_g9",
    "lucario_gold_exact",
    "alakazam_large_g9",
    "kangaskhan_crustle_large_g9",
    "festival_grass_large_g9",
    "mega_lopunny_large_g49",
    "teal_mask_ogerpon_large_g51",
    "arboliva_meganium_ogerpon_large_g51",
    "universal_ppo_large_256x6",
]
HISTORY_GENERATIONS = 10
EXTERNAL_CACHE_SCHEMA = 2


def chain_is_active(chain_name: str, chain: dict) -> bool:
    training = chain.get("trainingControl", {})
    retirement = chain.get("retirement", {})
    retirement_status = str(retirement.get("status", ""))
    rollout_enabled = training.get("rollout", {}).get("enabled") is True
    # Older active chain records predate the explicit learner.enabled flag.
    # Absence means enabled; an explicit false always wins.
    learner_enabled = training.get("learner", {}).get("enabled", True) is True
    paused = training.get("paused") is True
    if chain_name == "universal_ppo_large_256x6":
        return retirement_status != "retired_disabled" and learner_enabled and not paused
    return (
        retirement_status != "retired_disabled"
        and rollout_enabled
        and learner_enabled
        and not paused
    )


def active_chain_names(league_root: Path) -> list[str]:
    league = read_object(league_root / "state" / "league.json")
    return [
        name
        for name, chain in league.get("chains", {}).items()
        if chain_is_active(name, chain)
    ]


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def collect_rollout_summaries(league_root: Path, chain: str) -> dict:
    grouped = {}
    seen_outputs = set()
    for path in (league_root / "buffer" / "ready" / chain).glob(
        "*.jsonl.gz.summary.json"
    ):
        try:
            row = read_object(path)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        output = row.get("output", {})
        output_key = (str(output.get("path", "")), str(output.get("sha256", "")))
        if output_key in seen_outputs:
            continue
        seen_outputs.add(output_key)
        try:
            generation = int(row.get("behaviorGeneration", -1))
        except (TypeError, ValueError):
            continue
        snapshot_id = str(row.get("behaviorSnapshotId", ""))
        if generation < 0 or not snapshot_id:
            continue
        grouped.setdefault((generation, snapshot_id), []).append(row)
    return grouped


def published_snapshots(
    league_root: Path, chain: str, first_generation: int, last_generation: int
) -> dict:
    selected = {}
    learner_root = league_root / "learners" / chain
    for path in learner_root.glob("generation-*/PUBLISHED.json"):
        try:
            row = read_object(path)
            generation = int(row.get("generation", -1))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not first_generation <= generation <= last_generation:
            continue
        snapshot_id = str(row.get("snapshotId", ""))
        if not snapshot_id:
            continue
        modified = path.stat().st_mtime_ns
        previous = selected.get(generation)
        if previous is None or modified > previous[0]:
            selected[generation] = (modified, snapshot_id)
    return {generation: value[1] for generation, value in selected.items()}


def generation_external(selected: list, generation: int, snapshot_id: str) -> dict:
    patterns = {
        "episode": re.compile(rb'"episode_id":"([^"]+)"'),
        "player": re.compile(rb'"player":([01])'),
        "selfPlay": re.compile(rb'"self_play":(true|false)'),
        "outcome": re.compile(rb'"outcome":(-?[0-9]+(?:\.[0-9]+)?)'),
    }
    episodes = {}
    parse_errors = 0
    conflicted_episode_ids = set()
    for summary in selected:
        rollout_path = Path(str(summary.get("output", {}).get("path", "")))
        try:
            handle = gzip.open(rollout_path, "rb")
        except OSError:
            parse_errors += 1
            continue
        with handle:
            for line in handle:
                matches = {key: pattern.search(line) for key, pattern in patterns.items()}
                if any(match is None for match in matches.values()):
                    parse_errors += 1
                    continue
                episode_id = matches["episode"].group(1).decode("utf-8")
                value = (
                    matches["selfPlay"].group(1) == b"true",
                    int(matches["player"].group(1)),
                    float(matches["outcome"].group(1)),
                )
                previous = episodes.setdefault(episode_id, value)
                if previous != value:
                    conflicted_episode_ids.add(episode_id)
    clean_episodes = {
        episode_id: value
        for episode_id, value in episodes.items()
        if episode_id not in conflicted_episode_ids
    }
    external_rows = [value for value in clean_episodes.values() if not value[0]]
    wins = sum(value[2] > 0 for value in external_rows)
    losses = sum(value[2] < 0 for value in external_rows)
    draws = sum(value[2] == 0 for value in external_rows)
    external = len(external_rows)
    seat = {}
    for player in (0, 1):
        rows = [value for value in external_rows if value[1] == player]
        seat[str(player)] = {
            "episodes": len(rows),
            "wins": sum(value[2] > 0 for value in rows),
            "losses": sum(value[2] < 0 for value in rows),
            "draws": sum(value[2] == 0 for value in rows),
        }
        seat[str(player)]["winRate"] = (
            seat[str(player)]["wins"] / len(rows) if rows else None
        )
    summary_wins = sum(int(row.get("wins", 0)) for row in selected)
    summary_losses = sum(int(row.get("losses", 0)) for row in selected)
    summary_draws = sum(int(row.get("draws", 0)) for row in selected)
    return {
        "generation": generation,
        "snapshotId": snapshot_id,
        "completedShards": len(selected),
        "externalEpisodes": external,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "winRate": wins / external if external else None,
        "seat": seat,
        "selfPlayEpisodes": sum(value[0] for value in clean_episodes.values()),
        "latestShardAt": max(
            (str(row.get("createdAt", "")) for row in selected), default=None
        ),
        "dedupe": (
            "episodeId across unique rollout output path+sha256; conflicting "
            "episodeIds excluded from all W-L/seat counts"
        ),
        "aggregationScope": (
            "single behaviorGeneration+behaviorSnapshotId; never cumulative"
        ),
        "episodeParseErrors": parse_errors,
        "episodeConflicts": len(conflicted_episode_ids),
        "excludedConflictEpisodes": len(conflicted_episode_ids),
        "summaryParity": {
            "wins": summary_wins,
            "losses": summary_losses,
            "draws": summary_draws,
            "matches": (wins, losses, draws)
            == (summary_wins, summary_losses, summary_draws),
        },
    }


def external_history(
    league_root: Path,
    chain: str,
    current_generation: int,
    current_snapshot_id: str,
    limit: int = HISTORY_GENERATIONS,
) -> list:
    first_generation = max(0, current_generation - limit + 1)
    grouped = collect_rollout_summaries(league_root, chain)
    snapshots = published_snapshots(
        league_root, chain, first_generation, current_generation
    )
    snapshots[current_generation] = current_snapshot_id
    result = []
    cache_root = (
        league_root
        / "monitoring"
        / "gold-acceleration"
        / "generation-external-cache"
        / chain
    )
    for generation in range(first_generation, current_generation + 1):
        snapshot_id = snapshots.get(generation)
        if snapshot_id is None:
            candidates = [
                (key[1], rows)
                for key, rows in grouped.items()
                if key[0] == generation
            ]
            if candidates:
                snapshot_id = max(
                    candidates,
                    key=lambda item: (
                        len(item[1]),
                        max(str(row.get("createdAt", "")) for row in item[1]),
                    ),
                )[0]
        cache_path = cache_root / f"generation-{generation:06d}.json"
        cached = None
        if generation < current_generation and cache_path.is_file():
            try:
                candidate = read_object(cache_path)
                if (
                    candidate.get("cacheSchemaVersion") == EXTERNAL_CACHE_SCHEMA
                    and candidate.get("snapshotId") == (snapshot_id or "")
                ):
                    cached = candidate
            except (OSError, json.JSONDecodeError, TypeError):
                cached = None
        if cached is not None:
            result.append(cached)
            continue
        selected = grouped.get((generation, snapshot_id), []) if snapshot_id else []
        row = generation_external(selected, generation, snapshot_id or "")
        row["cacheSchemaVersion"] = EXTERNAL_CACHE_SCHEMA
        result.append(row)
        if generation < current_generation:
            cache_root.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            os.replace(temporary, cache_path)
    for index, row in enumerate(result):
        previous_rate = result[index - 1].get("winRate") if index else None
        current_rate = row.get("winRate")
        row["externalWinRateDeltaPp"] = (
            (current_rate - previous_rate) * 100.0
            if current_rate is not None and previous_rate is not None
            else None
        )
        if row.get("episodeParseErrors", 0):
            row["sampleStatus"] = "conflicted"
        elif row.get("externalEpisodes", 0) >= 40:
            row["sampleStatus"] = "sufficient"
        elif row.get("externalEpisodes", 0) > 0:
            row["sampleStatus"] = "locating_only"
        else:
            row["sampleStatus"] = "waiting"
        row["dataQuality"] = (
            "conflicts_excluded"
            if row.get("excludedConflictEpisodes", 0)
            else "clean"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--league-root", type=Path)
    args = parser.parse_args()
    data = json.load(sys.stdin)
    fields = [
        "generation",
        "snapshotId",
        "completedShards",
        "episodes",
        "decisions",
        "selfPlayEpisodes",
        "livePpoOpponentEpisodes",
        "publishedUpdates",
        "failedUpdates",
        "latestInitialPolicyShift",
        "latestEpoch",
    ]
    result = {
        "leagueUpdatedAt": data.get("leagueUpdatedAt"),
        "poolSha256": data.get("poolSha256"),
        "liveAgents": [
            {
                "chain": agent.get("chain"),
                "generation": agent.get("generation"),
                "name": agent.get("name"),
            }
            for agent in data.get("liveAgents", [])
        ],
        "chains": {},
    }
    chains = active_chain_names(args.league_root) if args.league_root is not None else CHAINS
    for chain in chains:
        source = data.get("chains", {}).get(chain, {})
        result["chains"][chain] = {field: source.get(field) for field in fields}
        if args.league_root is not None and source.get("generation") is not None:
            history = external_history(
                args.league_root,
                chain,
                int(source["generation"]),
                str(source.get("snapshotId", "")),
            )
            result["chains"][chain]["generationExternalHistory"] = history
            result["chains"][chain]["currentGenerationExternal"] = history[-1]
    rendered = json.dumps(result, ensure_ascii=False)
    if args.output is not None:
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, output)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
