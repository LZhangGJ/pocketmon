from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
HIGH_CONFIDENCE_FLAGS = (
    "declined_optional_selection",
    "ended_with_attach_available",
    "ended_with_evolve_available",
)
FLAG_LABELS = {
    "declined_optional_selection": "可选检索选空",
    "ended_with_attach_available": "可贴能量却结束回合",
    "ended_with_evolve_available": "可进化却结束回合",
    "attacked_before_energy_attachment": "攻击前仍可贴能量",
    "attacked_with_ability_available": "攻击时仍有能力可用",
    "attacked_with_evolve_available": "攻击时仍可进化",
    "attacked_with_play_available": "攻击时仍有手牌可打",
}


def archetype(signature: str) -> str:
    if "Marnie's Grimmsnarl" in signature:
        return "Grimmsnarl 镜像"
    if "Alakazam" in signature:
        return "Alakazam"
    if "Mega Lopunny" in signature:
        return "Mega Lopunny"
    if "Mega Lucario" in signature:
        return "Mega Lucario"
    if "Dragapult" in signature:
        return "Dragapult"
    if "Mega Kangaskhan" in signature:
        return "Mega Kangaskhan"
    if "Mega Starmie" in signature:
        return "Mega Starmie"
    if any(
        name in signature
        for name in ("Rillaboom", "Thwackey", "Dipplin", "Teal Mask Ogerpon", "Brambleghast")
    ):
        return "Grass / Ogerpon"
    return "其他"


def public_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if "PUBLIC" in str(row["episode_type"])]


def main() -> None:
    episodes = json.loads((OUTPUT / "episodes.json").read_text(encoding="utf-8"))
    decisions = json.loads((OUTPUT / "loss_decisions.json").read_text(encoding="utf-8"))
    public = public_rows(episodes)
    losses = [row for row in public if row["result"] == "loss"]

    matchups: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in public:
        matchups[archetype(row["opponent_deck"])][row["result"]] += 1
    matchup_rows = []
    for name, counts in matchups.items():
        games = counts["win"] + counts["loss"]
        matchup_rows.append(
            {
                "archetype": name,
                "games": games,
                "wins": counts["win"],
                "losses": counts["loss"],
                "win_rate": counts["win"] / games if games else 0.0,
                "loss_share": counts["loss"] / len(losses) if losses else 0.0,
            }
        )
    matchup_rows.sort(key=lambda row: (-row["games"], row["archetype"]))

    all_flags = sorted({flag for row in public for flag in row["flags"]})
    flag_rows = []
    for flag in all_flags:
        loss_episodes = sum(flag in row["flags"] for row in losses)
        wins = [row for row in public if row["result"] == "win"]
        win_episodes = sum(flag in row["flags"] for row in wins)
        loss_rate = loss_episodes / len(losses)
        win_rate = win_episodes / len(wins)
        flag_rows.append(
            {
                "flag": flag,
                "label": FLAG_LABELS.get(flag, flag),
                "loss_episodes": loss_episodes,
                "loss_episode_rate": loss_rate,
                "win_episodes": win_episodes,
                "win_episode_rate": win_rate,
                "loss_minus_win_pp": (loss_rate - win_rate) * 100,
                "high_confidence": flag in HIGH_CONFIDENCE_FLAGS,
            }
        )
    flag_rows.sort(key=lambda row: (-row["loss_minus_win_pp"], row["flag"]))

    public_loss_decisions = [
        row for row in decisions if "PUBLIC" in str(row["episode_type"])
    ]
    events_by_episode: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for decision in public_loss_decisions:
        for flag in decision["flags"]:
            if flag not in HIGH_CONFIDENCE_FLAGS:
                continue
            events_by_episode[int(decision["episode_id"])].append(
                {
                    "flag": flag,
                    "turn": decision["state"]["turn"],
                    "action_step": decision["action_step"],
                    "available": decision["available"],
                    "self_active": (decision["state"]["self"].get("active") or {}).get("name"),
                    "opponent_active": (decision["state"]["opponent"].get("active") or {}).get("name"),
                }
            )

    loss_by_id = {int(row["episode_id"]): row for row in losses}
    reviewed_losses = []
    for episode_id, events in sorted(events_by_episode.items()):
        episode = loss_by_id[episode_id]
        counts = Counter(event["flag"] for event in events)
        turns = defaultdict(list)
        for event in events:
            turns[event["flag"]].append(int(event["turn"] or 0))
        severity = "高" if len(events) >= 2 or any(1 in values for values in turns.values()) else "中"
        reviewed_losses.append(
            {
                "episode_id": episode_id,
                "opponent": episode["opponent"],
                "opponent_deck": episode["opponent_deck"],
                "archetype": archetype(episode["opponent_deck"]),
                "max_turn": episode["max_turn"],
                "severity": severity,
                "event_count": len(events),
                "flags": dict(counts),
                "turns": {key: values for key, values in turns.items()},
                "events": events,
            }
        )
    reviewed_losses.sort(key=lambda row: (-row["event_count"], row["episode_id"]))

    wins = [row for row in public if row["result"] == "win"]
    affected_losses = sum(
        any(flag in row["flags"] for flag in HIGH_CONFIDENCE_FLAGS) for row in losses
    )
    affected_wins = sum(
        any(flag in row["flags"] for flag in HIGH_CONFIDENCE_FLAGS) for row in wins
    )
    event_counts = Counter(
        event["flag"] for events in events_by_episode.values() for event in events
    )
    event_episode_counts = Counter(
        flag
        for row in losses
        for flag in HIGH_CONFIDENCE_FLAGS
        if flag in row["flags"]
    )

    payload = {
        "snapshot_date": "2026-08-09",
        "submission_id": 55328694,
        "public_games": len(public),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(public),
        "high_confidence_affected_loss_episodes": affected_losses,
        "high_confidence_affected_loss_rate": affected_losses / len(losses),
        "comparison_affected_win_episodes": affected_wins,
        "comparison_affected_win_rate": affected_wins / len(wins),
        "high_confidence_event_counts": dict(event_counts),
        "high_confidence_episode_counts": dict(event_episode_counts),
        "matchups": matchup_rows,
        "flags": flag_rows,
        "reviewed_losses": reviewed_losses,
    }
    (OUTPUT / "review_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (OUTPUT / "loss_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "episode_id",
            "severity",
            "opponent",
            "archetype",
            "opponent_deck",
            "max_turn",
            "event_count",
            "mistakes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in reviewed_losses:
            labels = []
            for flag in HIGH_CONFIDENCE_FLAGS:
                if flag in row["flags"]:
                    labels.append(
                        f"{FLAG_LABELS[flag]} x{row['flags'][flag]} (turn {','.join(map(str, row['turns'][flag]))})"
                    )
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields if field != "mistakes"},
                    "mistakes": "; ".join(labels),
                }
            )

    lines = [
        "# Submission 55328694 replay audit",
        "",
        f"Snapshot: 2026-08-09. Public games: {len(public)}; wins: {len(wins)}; losses: {len(losses)}; win rate: {len(wins) / len(public):.1%}.",
        "",
        f"High-confidence tactical-error indicators appeared in {affected_losses}/{len(losses)} losses ({affected_losses / len(losses):.1%}), versus {affected_wins}/{len(wins)} wins ({affected_wins / len(wins):.1%}). This is association, not proof that each event caused the result.",
        "",
        "## High-confidence error patterns",
        "",
    ]
    for flag in HIGH_CONFIDENCE_FLAGS:
        lines.append(
            f"- {FLAG_LABELS[flag]}: {event_counts[flag]} events across {event_episode_counts[flag]} public losses."
        )
    lines.extend(["", "## Matchups", "", "| Archetype | Games | W-L | Win rate |", "|---|---:|---:|---:|"])
    for row in matchup_rows:
        lines.append(
            f"| {row['archetype']} | {row['games']} | {row['wins']}-{row['losses']} | {row['win_rate']:.1%} |"
        )
    lines.extend(["", "## Flagged losses", "", "| Episode | Severity | Opponent archetype | Events |", "|---:|:---:|---|---|"])
    for row in reviewed_losses:
        labels = []
        for flag in HIGH_CONFIDENCE_FLAGS:
            if flag in row["flags"]:
                labels.append(
                    f"{FLAG_LABELS[flag]} x{row['flags'][flag]} (T{','.join(map(str, row['turns'][flag]))})"
                )
        lines.append(
            f"| {row['episode_id']} | {row['severity']} | {row['archetype']} | {'; '.join(labels)} |"
        )
    (OUTPUT / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({key: payload[key] for key in (
        "public_games",
        "wins",
        "losses",
        "win_rate",
        "high_confidence_affected_loss_episodes",
        "high_confidence_affected_loss_rate",
        "comparison_affected_win_episodes",
        "comparison_affected_win_rate",
        "high_confidence_event_counts",
        "high_confidence_episode_counts",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
