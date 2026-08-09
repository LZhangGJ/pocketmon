from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.public_replay import iter_transitions, terminal_outcome, validate_transition


FALLBACK_OPTION_NAMES = {
    0: "UNKNOWN",
    1: "YES",
    2: "NO",
    3: "CARD",
    4: "TOOL",
    5: "ENERGY",
    6: "POKEMON_ENERGY",
    7: "PLAY",
    8: "ATTACH",
    9: "EVOLVE",
    10: "ABILITY",
    11: "DISCARD",
    12: "RETREAT",
    13: "ATTACK",
    14: "END",
    15: "CARD_ID",
    16: "SPECIAL_CONDITION",
}


def enum_names(class_name: str, fallback: dict[int, str] | None = None) -> dict[int, str]:
    try:
        from cg import api as cg_api

        enum_class = getattr(cg_api, class_name)
        return {int(member.value): str(member.name) for member in enum_class}
    except (ImportError, AttributeError, TypeError, ValueError):
        return dict(fallback or {})


OPTION_NAMES = enum_names("OptionType", FALLBACK_OPTION_NAMES)
SELECT_CONTEXT_NAMES = enum_names("SelectContext")
SELECT_TYPE_NAMES = enum_names("SelectType")
AREA_NAMES = enum_names("AreaType")


def card_id(card: Any) -> int:
    return int(card.get("id", 0)) if isinstance(card, dict) else 0


def load_catalog(path: Path) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cards = {int(row["cardId"]): row for row in payload["cards"]}
    attacks = {int(row["attackId"]): row for row in payload["attacks"]}
    return cards, attacks


def card_name(card: Any, cards: dict[int, dict[str, Any]]) -> str:
    identifier = card_id(card)
    if isinstance(card, dict) and card.get("name"):
        return str(card["name"])
    if identifier in cards:
        return str(cards[identifier].get("name") or f"card:{identifier}")
    return f"card:{identifier}" if identifier else "none"


def zone_cards(observation: dict[str, Any], area: Any, index_owner: Any) -> list[Any]:
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    your_index = int(current.get("yourIndex", 0))
    owner = int(index_owner) if isinstance(index_owner, int) else your_index
    players = current.get("players") or []
    player = players[owner] if 0 <= owner < len(players) else {}
    return {
        1: select.get("deck") or [],
        2: player.get("hand") or [],
        3: player.get("discard") or [],
        4: player.get("active") or [],
        5: player.get("bench") or [],
        6: player.get("prize") or [],
        7: current.get("stadium") or [],
        12: current.get("looking") or [],
    }.get(area, [])


def resolve_area_card(
    observation: dict[str, Any], area: Any, index: Any, owner: Any = None
) -> dict[str, Any] | None:
    values = zone_cards(observation, area, owner)
    if isinstance(index, int) and 0 <= index < len(values) and isinstance(values[index], dict):
        return values[index]
    return None


def resolve_option_cards(
    observation: dict[str, Any], option: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    option_name = OPTION_NAMES.get(int(option.get("type", -1)), "UNKNOWN")
    current = observation.get("current") or {}
    your_index = int(current.get("yourIndex", 0))
    primary: dict[str, Any] | None = None
    target: dict[str, Any] | None = None
    if option_name == "PLAY":
        primary = resolve_area_card(observation, 2, option.get("index"), your_index)
    elif option_name in {"ATTACH", "EVOLVE"}:
        primary = resolve_area_card(
            observation, option.get("area"), option.get("index"), your_index
        )
        target = resolve_area_card(
            observation,
            option.get("inPlayArea"),
            option.get("inPlayIndex"),
            your_index,
        )
    elif option_name in {"ABILITY", "DISCARD", "RETREAT"}:
        primary = resolve_area_card(
            observation, option.get("area"), option.get("index"), your_index
        )
    elif option_name in {"CARD", "TOOL", "ENERGY", "POKEMON_ENERGY"}:
        target = resolve_area_card(
            observation,
            option.get("area"),
            option.get("index"),
            option.get("playerIndex"),
        )
        primary = target
        if option_name == "TOOL" and target is not None:
            children = target.get("tools") or []
            child_index = option.get("toolIndex")
            if isinstance(child_index, int) and 0 <= child_index < len(children):
                primary = children[child_index]
        elif option_name in {"ENERGY", "POKEMON_ENERGY"} and target is not None:
            children = target.get("energyCards") or []
            child_index = option.get("energyIndex")
            if isinstance(child_index, int) and 0 <= child_index < len(children):
                primary = children[child_index]
    elif isinstance(option.get("cardId"), int):
        primary = {"id": int(option["cardId"])}
    return primary, target


def option_description(
    observation: dict[str, Any],
    option: dict[str, Any],
    cards: dict[int, dict[str, Any]],
    attacks: dict[int, dict[str, Any]],
) -> str:
    option_type = int(option.get("type", -1))
    name = OPTION_NAMES.get(option_type, f"OPTION_{option_type}")
    primary, target = resolve_option_cards(observation, option)
    bits = [name]
    if primary is not None:
        bits.append(card_name(primary, cards))
    if target is not None and card_id(target) != card_id(primary):
        bits.append(f"to {card_name(target, cards)}")
    attack_id = option.get("attackId")
    if isinstance(attack_id, int):
        attack = attacks.get(attack_id, {})
        bits.append(str(attack.get("name") or f"attack:{attack_id}"))
    for key in ("number", "count", "specialConditionType"):
        if key in option:
            bits.append(f"{key}={option[key]}")
    return " | ".join(bits)


def pokemon_summary(card: Any, cards: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(card, dict):
        return None
    return {
        "name": card_name(card, cards),
        "id": card_id(card),
        "serial": card.get("serial"),
        "hp": card.get("hp"),
        "max_hp": card.get("maxHp"),
        "damage": (
            max(0, int(card.get("maxHp", 0)) - int(card.get("hp", 0)))
            if isinstance(card.get("hp"), (int, float))
            and isinstance(card.get("maxHp"), (int, float))
            else None
        ),
        "energies": list(card.get("energies") or []),
        "energy_cards": [card_name(row, cards) for row in (card.get("energyCards") or [])],
        "tools": [card_name(row, cards) for row in (card.get("tools") or [])],
    }


def state_summary(observation: dict[str, Any], cards: dict[int, dict[str, Any]]) -> dict[str, Any]:
    current = observation.get("current") or {}
    players = current.get("players") or []
    your_index = int(current.get("yourIndex", 0))
    opponent_index = 1 - your_index

    def player_summary(index: int, reveal_hand: bool) -> dict[str, Any]:
        player = players[index] if 0 <= index < len(players) else {}
        active = player.get("active") or []
        return {
            "active": pokemon_summary(active[0], cards) if active else None,
            "bench": [pokemon_summary(row, cards) for row in (player.get("bench") or [])],
            "prizes": len(player.get("prize") or []),
            "deck_count": player.get("deckCount"),
            "hand_count": player.get("handCount"),
            "hand": [card_name(row, cards) for row in (player.get("hand") or [])]
            if reveal_hand
            else [],
            "discard": [card_name(row, cards) for row in (player.get("discard") or [])],
        }

    return {
        "turn": current.get("turn"),
        "turn_action_count": current.get("turnActionCount"),
        "first_player": current.get("firstPlayer"),
        "energy_attached": bool(current.get("energyAttached")),
        "supporter_played": bool(current.get("supporterPlayed")),
        "stadium_played": bool(current.get("stadiumPlayed")),
        "retreated": bool(current.get("retreated")),
        "self": player_summary(your_index, True),
        "opponent": player_summary(opponent_index, False),
        "stadium": [card_name(row, cards) for row in (current.get("stadium") or [])],
    }


def deck_ids(replay: dict[str, Any], player: int) -> list[int]:
    try:
        visualize = replay["steps"][0][0]["visualize"]
        action = visualize[0]["action"]
        values = action[player]
        if isinstance(values, list) and len(values) == 60:
            return [int(value) for value in values]
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    for step in replay.get("steps") or []:
        for view in step if isinstance(step, list) else []:
            if not isinstance(view, dict):
                continue
            action = view.get("action")
            if isinstance(action, list) and len(action) == 60:
                return [int(value) for value in action]
    return []


def deck_signature(ids: list[int], cards: dict[int, dict[str, Any]]) -> str:
    counts = Counter(ids)
    pokemon = []
    for identifier, count in counts.items():
        metadata = cards.get(identifier, {})
        if metadata.get("cardType") == 0 or metadata.get("hp"):
            pokemon.append(
                (
                    int(bool(metadata.get("megaEx"))),
                    int(bool(metadata.get("ex"))),
                    int(bool(metadata.get("stage2"))),
                    count,
                    str(metadata.get("name") or f"card:{identifier}"),
                )
            )
    pokemon.sort(reverse=True)
    names = []
    for _, _, _, _, name in pokemon:
        if name not in names:
            names.append(name)
        if len(names) == 3:
            break
    return " / ".join(names) if names else "unknown deck"


def action_flags(observation: dict[str, Any], chosen_names: list[str]) -> list[str]:
    select = observation.get("select") or {}
    options = select.get("option") or []
    available = {OPTION_NAMES.get(int(row.get("type", -1)), "UNKNOWN") for row in options}
    current = observation.get("current") or {}
    flags: list[str] = []
    if "END" in chosen_names:
        if "ATTACK" in available:
            flags.append("ended_with_attack_available")
        if "ATTACH" in available and not current.get("energyAttached"):
            flags.append("ended_with_attach_available")
        if "ABILITY" in available:
            flags.append("ended_with_ability_available")
        if "EVOLVE" in available:
            flags.append("ended_with_evolve_available")
    if "ATTACK" in chosen_names:
        if "ATTACH" in available and not current.get("energyAttached"):
            flags.append("attacked_before_energy_attachment")
        if "ABILITY" in available:
            flags.append("attacked_with_ability_available")
        if "EVOLVE" in available:
            flags.append("attacked_with_evolve_available")
        if "PLAY" in available:
            flags.append("attacked_with_play_available")
    if not chosen_names and int(select.get("maxCount", 0) or 0) > 0:
        flags.append("declined_optional_selection")
    return flags


def replay_metadata_lookup(manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["id"]): row for row in manifest.get("episodes") or []}


def our_players(metadata: dict[str, Any], submission_id: int) -> list[int]:
    return sorted(
        int(agent["index"])
        for agent in metadata.get("agents") or []
        if int(agent.get("submission_id", -1)) == submission_id
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--submission-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cards, attacks = load_catalog(args.catalog.resolve())
    manifest = json.loads((input_dir / "episodes.json").read_text(encoding="utf-8"))
    metadata_by_id = replay_metadata_lookup(manifest)

    episode_rows: list[dict[str, Any]] = []
    loss_decisions: list[dict[str, Any]] = []
    invalid_decisions: list[dict[str, Any]] = []
    all_flag_counts: Counter[str] = Counter()
    loss_flag_counts: Counter[str] = Counter()
    matchup_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for position, path in enumerate(sorted((input_dir / "replays").glob("*.json")), start=1):
        replay = json.loads(path.read_text(encoding="utf-8"))
        episode_id = int((replay.get("info") or {}).get("EpisodeId"))
        metadata = metadata_by_id[episode_id]
        players = our_players(metadata, args.submission_id)
        terminal = terminal_outcome(replay)
        winner = terminal["winner"]
        episode_type = str(metadata.get("type", ""))
        team_names = list((replay.get("info") or {}).get("TeamNames") or [])

        for player in players:
            opponent = 1 - player
            result = "draw" if winner == 2 else "win" if winner == player else "loss"
            opponent_name = team_names[opponent] if opponent < len(team_names) else "unknown"
            own_signature = deck_signature(deck_ids(replay, player), cards)
            opponent_signature = deck_signature(deck_ids(replay, opponent), cards)
            matchup_counts[opponent_signature][result] += 1

            decision_count = 0
            model_decision_count = 0
            forced_decision_count = 0
            episode_flags: Counter[str] = Counter()
            option_type_counts: Counter[str] = Counter()
            max_turn = 0
            final_state: dict[str, Any] | None = None
            for transition in iter_transitions(replay, "previous"):
                if transition.player != player:
                    continue
                validation = validate_transition(transition)
                if not validation.valid:
                    invalid_decisions.append(
                        {
                            "episode_id": episode_id,
                            "player": player,
                            "action_step": transition.action_step,
                            "reason": validation.reason,
                            "action": transition.action,
                        }
                    )
                    continue
                if validation.kind != "decision" or transition.observation is None:
                    continue
                decision_count += 1
                observation = transition.observation
                select = observation.get("select") or {}
                options = select.get("option") or []
                chosen_indices = list(transition.action or [])
                chosen = [options[index] for index in chosen_indices]
                chosen_names = [
                    OPTION_NAMES.get(int(option.get("type", -1)), "UNKNOWN") for option in chosen
                ]
                for name in chosen_names:
                    option_type_counts[name] += 1
                if int(select.get("minCount", 0) or 0) == int(select.get("maxCount", 0) or 0) and int(
                    select.get("minCount", 0) or 0
                ) in (0, len(options)):
                    forced_decision_count += 1
                else:
                    model_decision_count += 1
                flags = action_flags(observation, chosen_names)
                episode_flags.update(flags)
                all_flag_counts.update(flags)
                state = state_summary(observation, cards)
                final_state = state
                max_turn = max(max_turn, int(state.get("turn") or 0))
                if result == "loss":
                    loss_flag_counts.update(flags)
                    loss_decisions.append(
                        {
                            "episode_id": episode_id,
                            "episode_type": episode_type,
                            "player": player,
                            "opponent": opponent_name,
                            "opponent_deck": opponent_signature,
                            "action_step": transition.action_step,
                            "select_type": int(select.get("type", -1)),
                            "select_type_name": SELECT_TYPE_NAMES.get(
                                int(select.get("type", -1)), ""
                            ),
                            "select_context": int(select.get("context", -1)),
                            "select_context_name": SELECT_CONTEXT_NAMES.get(
                                int(select.get("context", -1)), ""
                            ),
                            "chosen_indices": chosen_indices,
                            "chosen": [
                                option_description(observation, row, cards, attacks)
                                for row in chosen
                            ],
                            "available": [
                                option_description(observation, row, cards, attacks)
                                for row in options
                            ],
                            "flags": flags,
                            "state": state,
                        }
                    )

            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "episode_type": episode_type,
                    "create_time": metadata.get("create_time"),
                    "player": player,
                    "opponent": opponent_name,
                    "result": result,
                    "winner": winner,
                    "our_deck": own_signature,
                    "opponent_deck": opponent_signature,
                    "seed": (replay.get("configuration") or {}).get("seed"),
                    "steps": len(replay.get("steps") or []),
                    "max_turn": max_turn,
                    "decisions": decision_count,
                    "model_decisions": model_decision_count,
                    "forced_decisions": forced_decision_count,
                    "flag_count": sum(episode_flags.values()),
                    "flags": dict(episode_flags),
                    "chosen_option_types": dict(option_type_counts),
                    "final_state": final_state,
                    "replay_file": path.name,
                }
            )
        print(f"replays={position}/{manifest['episode_count']} id={episode_id}", flush=True)

    public_rows = [row for row in episode_rows if "PUBLIC" in row["episode_type"]]
    public_outcomes = Counter(row["result"] for row in public_rows)
    summary = {
        "submission_id": args.submission_id,
        "manifest_episode_count": manifest.get("episode_count"),
        "replay_files": len(list((input_dir / "replays").glob("*.json"))),
        "analyzed_player_episodes": len(episode_rows),
        "public_player_episodes": len(public_rows),
        "public_outcomes": dict(public_outcomes),
        "public_win_rate": (
            public_outcomes["win"] / len(public_rows) if public_rows else None
        ),
        "validation_player_episodes": len(episode_rows) - len(public_rows),
        "invalid_decisions": len(invalid_decisions),
        "all_candidate_flag_counts": dict(all_flag_counts),
        "loss_candidate_flag_counts": dict(loss_flag_counts),
        "option_names": OPTION_NAMES,
        "select_context_names": SELECT_CONTEXT_NAMES,
        "matchups": {
            matchup: dict(counts)
            for matchup, counts in sorted(matchup_counts.items())
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "episodes.json").write_text(
        json.dumps(episode_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "loss_decisions.json").write_text(
        json.dumps(loss_decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "invalid_decisions.json").write_text(
        json.dumps(invalid_decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "episodes.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "episode_id",
            "episode_type",
            "create_time",
            "player",
            "opponent",
            "result",
            "our_deck",
            "opponent_deck",
            "seed",
            "steps",
            "max_turn",
            "decisions",
            "model_decisions",
            "forced_decisions",
            "flag_count",
            "replay_file",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in episode_rows:
            writer.writerow({key: row.get(key) for key in fields})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
