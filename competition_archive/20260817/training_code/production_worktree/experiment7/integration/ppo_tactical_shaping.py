from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


TYPE_PLAY = 7
TYPE_ATTACH = 8
TYPE_EVOLVE = 9
TYPE_ATTACK = 13
TYPE_END = 14
A08_TERMINAL_BEFORE_EVOLVE_MODES = frozenset({"control", "end_only", "gated"})
A08_MAXIMUM_BELT_ATTACKERS = frozenset({"Applin", "Dipplin"})
A08_MAXIMUM_BELT_SUPPORTS = frozenset({"Grookey", "Thwackey"})
TACTICAL_ERROR_TO_OPPORTUNITY = {
    "a02_ended_with_damaging_attack_available": "a02_damaging_attack_before_end",
    "a08_ended_with_damaging_attack_available": "a08_damaging_attack_before_end",
    "lucario_ended_with_damaging_attack_available": "lucario_damaging_attack_before_end",
    "a02_declined_snorunt_after_poffin_with_froslass_in_hand": "a02_poffin_snorunt_setup",
    "a02_terminal_before_second_grimmsnarl": "a02_second_grimmsnarl_before_terminal",
    "a02_nonlethal_attack_before_second_grimmsnarl": "a02_second_grimmsnarl_before_terminal",
    "a02_terminal_before_froslass_setup": "a02_froslass_setup_before_terminal",
    "a02_nonlethal_attack_before_froslass_setup": "a02_froslass_setup_before_terminal",
    "a02_terminal_before_grimmsnarl_setup": "a02_grimmsnarl_setup_before_terminal",
    "a02_nonlethal_attack_before_grimmsnarl_setup": "a02_grimmsnarl_setup_before_terminal",
    "a02_terminal_before_manual_attach": "a02_manual_attach_before_terminal",
    "a02_nonlethal_attack_before_manual_attach": "a02_manual_attach_before_terminal",
    "a02_terminal_before_boss_low_hp_bench": "a02_boss_low_hp_bench_before_terminal",
    "a02_nonlethal_attack_before_boss_low_hp_bench": "a02_boss_low_hp_bench_before_terminal",
    "consumed_supporter_before_boss_low_hp_bench": "a02_boss_reserved_before_other_supporter",
    "a02_ended_after_boss_with_attack_available": "a02_attack_follow_through_after_boss",
    "a02_overfilled_munkidori_bench": "a02_projected_bench_budget",
    "a08_terminal_before_evolve": "a08_high_value_evolve_before_terminal",
    "a08_nonlethal_attack_before_high_value_evolve": "a08_high_value_evolve_before_nonlethal_attack",
    "a08_ended_before_board_wipe_recovery": "a08_board_wipe_recovery",
    "a08_ended_before_second_attacker_setup": "a08_second_attacker_setup",
    "lucario_end_before_high_value_evolve": "lucario_high_value_evolve_before_terminal",
    "lucario_nonlethal_attack_before_high_value_evolve": "lucario_high_value_evolve_before_terminal",
    "lucario_end_before_manual_attach": "lucario_attach_before_terminal",
    "lucario_nonlethal_attack_before_manual_attach": "lucario_attach_before_terminal",
    "lucario_mega_brave_overkill_forfeited_aura_acceleration": "lucario_aura_jab_lethal_acceleration",
    "dragapult_overstayed_budew_after_attack_line_ready": "dragapult_budew_launch_budget",
    "dragapult_delayed_ready_attacker_with_stall_action": "dragapult_ready_attacker_takeover",
    "dragapult_terminal_before_safe_dragapult_evolution": "dragapult_safe_evolution_before_terminal",
    "dragapult_nonprize_stall_before_safe_dragapult_evolution": "dragapult_safe_evolution_before_terminal",
    "dragapult_played_low_value_bench_filler_into_wall": "dragapult_wall_resource_budget",
    "dragapult_searched_thin_deck_into_wall": "dragapult_wall_resource_budget",
    "dragapult_consumed_supporter_before_wall_boss": "dragapult_ex_wall_bypass",
}


@dataclass(frozen=True)
class TacticalAdjustment:
    reward: float
    events: tuple[str, ...]
    preferred_action: tuple[int, ...] = ()
    deferred_attack_penalty: float = 0.0
    deferred_preferred_action: tuple[int, ...] = ()
    evolve_target_active: bool = False
    deferred_attack_events: tuple[str, ...] = ()
    opportunities: tuple[str, ...] = ()


def finalize_deferred_adjustment(
    adjustment: TacticalAdjustment,
    *,
    ko: bool,
    prize_delta: int,
    terminal_after_action: bool,
) -> TacticalAdjustment:
    """Resolve a gated attack after its observable engine effects are known."""

    penalty = float(adjustment.deferred_attack_penalty)
    if penalty <= 0 or ko or prize_delta > 0 or terminal_after_action:
        return TacticalAdjustment(
            adjustment.reward,
            adjustment.events,
            adjustment.preferred_action,
            evolve_target_active=adjustment.evolve_target_active,
            opportunities=adjustment.opportunities,
        )
    deferred_events = adjustment.deferred_attack_events or (
        "a08_nonlethal_attack_before_high_value_evolve",
    )
    return TacticalAdjustment(
        adjustment.reward - penalty,
        (*adjustment.events, *deferred_events),
        adjustment.deferred_preferred_action,
        evolve_target_active=adjustment.evolve_target_active,
        opportunities=adjustment.opportunities,
    )


def finalize_tactical_audit_row(row: dict[str, Any], *, final_win: bool) -> None:
    """Resolve deferred shaping and leave only JSON-serializable audit fields."""

    row.setdefault("ko", False)
    row.setdefault("prize_delta", 0)
    row.setdefault("terminal_after_action", False)
    row.setdefault("evolve_target_active", False)
    row["final_win"] = bool(final_win)
    adjustment = TacticalAdjustment(
        float(row.get("tactical_reward", 0.0)),
        tuple(row.get("tactical_events", [])),
        tuple(row.get("tactical_preferred_action", [])),
        deferred_attack_penalty=float(
            row.pop("_tactical_deferred_attack_penalty", 0.0)
        ),
        deferred_preferred_action=tuple(
            row.pop("_tactical_deferred_preferred_action", [])
        ),
        evolve_target_active=bool(row["evolve_target_active"]),
        deferred_attack_events=tuple(
            row.pop("_tactical_deferred_attack_events", [])
        ),
        opportunities=tuple(row.get("tactical_opportunities", [])),
    )
    adjustment = finalize_deferred_adjustment(
        adjustment,
        ko=bool(row["ko"]),
        prize_delta=int(row["prize_delta"]),
        terminal_after_action=bool(row["terminal_after_action"]),
    )
    row["tactical_reward"] = adjustment.reward
    row["tactical_events"] = list(adjustment.events)
    row["tactical_preferred_action"] = list(adjustment.preferred_action)
    row["tactical_opportunities"] = list(adjustment.opportunities)


@dataclass
class TacticalShapingState:
    """Episode-local state for tactical signals that span multiple decisions."""

    boss_reservation_turns: dict[int, int] = field(default_factory=dict)
    boss_post_play_turns: dict[int, int] = field(default_factory=dict)
    attack_counts: Counter[tuple[int, str]] = field(default_factory=Counter)


def _card_id(card: Any) -> int:
    return int(card.get("id", 0)) if isinstance(card, dict) else 0


def _card_name(card: Any, cards: dict[int, dict[str, Any]]) -> str:
    metadata = cards.get(_card_id(card), {})
    return str(metadata.get("name") or "")


def _option_cards(
    observation: dict[str, Any], option: dict[str, Any], features: Any
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    return features.resolve_option_cards(observation, option)


def _own_board(observation: dict[str, Any]) -> list[dict[str, Any]]:
    current = observation.get("current") or {}
    players = current.get("players") or []
    actor = int(current.get("yourIndex", 0) or 0)
    if not 0 <= actor < len(players) or not isinstance(players[actor], dict):
        return []
    player = players[actor]
    return [
        row
        for row in [*(player.get("active") or []), *(player.get("bench") or [])]
        if isinstance(row, dict)
    ]


def _opponent_board(observation: dict[str, Any]) -> list[dict[str, Any]]:
    current = observation.get("current") or {}
    players = current.get("players") or []
    actor = int(current.get("yourIndex", 0) or 0)
    opponent = 1 - actor
    if not 0 <= opponent < len(players) or not isinstance(players[opponent], dict):
        return []
    player = players[opponent]
    return [
        row
        for row in [*(player.get("active") or []), *(player.get("bench") or [])]
        if isinstance(row, dict)
    ]


def _option_name(
    observation: dict[str, Any],
    option: dict[str, Any],
    features: Any,
    cards: dict[int, dict[str, Any]],
) -> str:
    primary, _ = _option_cards(observation, option, features)
    return _card_name(primary, cards)


def _is_high_value_evolution(
    observation: dict[str, Any],
    option: dict[str, Any],
    features: Any,
    cards: dict[int, dict[str, Any]],
) -> bool:
    """Return whether a legal evolution merits gated attack ordering."""

    if int(option.get("inPlayArea", -1)) == 4:
        return True
    primary, target = _option_cards(observation, option, features)
    metadata = cards.get(_card_id(primary), {})
    if metadata.get("skills"):
        return True
    if any(bool(metadata.get(key)) for key in ("stage2", "ex", "megaEx", "tera")):
        return True
    evolved_hp = int(metadata.get("hp", 0) or 0)
    target_max_hp = int((target or {}).get("maxHp", 0) or 0)
    return target_max_hp > 0 and evolved_hp - target_max_hp >= 30


def _is_boss_name(name: str) -> bool:
    return "Boss's Orders" in name.replace("\N{RIGHT SINGLE QUOTATION MARK}", "'")


def _is_supporter(card: Any, cards: dict[int, dict[str, Any]]) -> bool:
    return int(cards.get(_card_id(card), {}).get("cardType", -1)) == 3


def _maximum_base_attack_damage(
    options: list[dict[str, Any]], attacks: dict[int, dict[str, Any]]
) -> int:
    damages = []
    for option in options:
        if int(option.get("type", -1)) != TYPE_ATTACK:
            continue
        attack_id = option.get("attackId")
        if not isinstance(attack_id, int):
            continue
        damage = attacks.get(attack_id, {}).get("damage", 0)
        if isinstance(damage, (int, float)):
            damages.append(max(int(damage), 0))
    return max(damages, default=0)


def _first_option_of_type(options: list[dict[str, Any]], option_type: int) -> int | None:
    return next(
        (index for index, option in enumerate(options) if int(option.get("type", -1)) == option_type),
        None,
    )


def _attached_energy_count(card: dict[str, Any] | None) -> int:
    if not isinstance(card, dict):
        return 0
    for key in ("energyCards", "energies", "attachedEnergy"):
        value = card.get(key)
        if isinstance(value, list):
            return len(value)
        if isinstance(value, (int, float)):
            return max(int(value), 0)
    return 0


def _attack_option_by_name(
    options: list[dict[str, Any]],
    attacks: dict[int, dict[str, Any]],
    name: str,
) -> tuple[int, dict[str, Any], dict[str, Any]] | None:
    for index, option in enumerate(options):
        if int(option.get("type", -1)) != TYPE_ATTACK:
            continue
        attack_id = option.get("attackId")
        if not isinstance(attack_id, int):
            continue
        attack = attacks.get(attack_id, {})
        if str(attack.get("name") or "") == name:
            return index, option, attack
    return None


def _lucario_aura_acceleration_available(
    observation: dict[str, Any], cards: dict[int, dict[str, Any]]
) -> bool:
    """Return whether Aura Jab has a real undercharged-bench target and fuel."""

    current = observation.get("current") or {}
    players = current.get("players") or []
    actor = int(current.get("yourIndex", 0) or 0)
    if not 0 <= actor < len(players) or not isinstance(players[actor], dict):
        return False
    player = players[actor]
    bench = [row for row in player.get("bench") or [] if isinstance(row, dict)]
    if not any(_attached_energy_count(row) < 2 for row in bench):
        return False
    for card in player.get("discard") or []:
        if not isinstance(card, dict):
            continue
        metadata = cards.get(_card_id(card), {})
        if int(metadata.get("cardType", -1)) == 5 and int(
            metadata.get("energyType", -1)
        ) == 6:
            return True
    return False


def _opponent_active_hp(observation: dict[str, Any]) -> int | None:
    current = observation.get("current") or {}
    players = current.get("players") or []
    actor = int(current.get("yourIndex", 0) or 0)
    opponent = 1 - actor
    if not 0 <= opponent < len(players) or not isinstance(players[opponent], dict):
        return None
    active = players[opponent].get("active") or []
    if not active or not isinstance(active[0], dict):
        return None
    hp = active[0].get("hp")
    return max(int(hp), 0) if isinstance(hp, (int, float)) else None


def _active_name(
    observation: dict[str, Any], cards: dict[int, dict[str, Any]], *, opponent: bool
) -> str:
    board = _opponent_board(observation) if opponent else _own_board(observation)
    return _card_name(board[0], cards) if board else ""


def _active_is_ex_wall(
    observation: dict[str, Any], cards: dict[int, dict[str, Any]]
) -> bool:
    """Recognize engine cards that prevent attack damage from opposing Pokémon ex."""

    board = _opponent_board(observation)
    if not board:
        return False
    metadata = cards.get(_card_id(board[0]), {})
    for skill in metadata.get("skills") or []:
        if not isinstance(skill, dict):
            continue
        text = str(skill.get("text") or "").lower()
        normalized = text.replace("pokémon", "pokemon")
        if (
            "prevent all damage" in normalized
            and "opponent" in normalized
            and ("{ex}" in normalized or "pokemon ex" in normalized)
        ):
            return True
    return False


def _own_active_is_ex(
    observation: dict[str, Any], cards: dict[int, dict[str, Any]]
) -> bool:
    board = _own_board(observation)
    if not board:
        return False
    metadata = cards.get(_card_id(board[0]), {})
    return bool(metadata.get("ex") or metadata.get("megaEx"))


def _has_attack_energy(card: dict[str, Any], attack: dict[str, Any]) -> bool:
    attached = card.get("energies")
    required = attack.get("energies")
    if not isinstance(attached, list) or not isinstance(required, list):
        return False
    attached_counts = Counter(int(value) for value in attached)
    colorless = int(attached_counts.get(0, 0))
    for energy in (int(value) for value in required if int(value) != 0):
        if attached_counts.get(energy, 0) > 0:
            attached_counts[energy] -= 1
        else:
            return False
    required_colorless = sum(int(value) == 0 for value in required)
    remaining = colorless + sum(attached_counts.values())
    return remaining >= required_colorless


def _successor_attach_option(
    observation: dict[str, Any],
    options: list[dict[str, Any]],
    features: Any,
    cards: dict[int, dict[str, Any]],
) -> int | None:
    """Prefer a viable bench successor over an already-ready/damaged active.

    This is deliberately a soft ranking only.  It does not infer dynamic attack
    costs or remove legal actions, and falls back to the first legal attachment
    when the engine does not expose enough target metadata.
    """

    board = _own_board(observation)
    active = board[0] if board else None
    active_serial = (active or {}).get("serial")
    candidates: list[tuple[tuple[int, int, int, int], int]] = []
    for index, option in enumerate(options):
        if int(option.get("type", -1)) != TYPE_ATTACH:
            continue
        _, target = _option_cards(observation, option, features)
        target = target if isinstance(target, dict) else {}
        name = _card_name(target, cards)
        serial = target.get("serial")
        is_active = bool(
            int(option.get("inPlayArea", -1)) == 4
            or (active_serial is not None and serial == active_serial)
        )
        hp = int(target.get("hp", 0) or 0)
        max_hp = int(target.get("maxHp", 0) or 0)
        active_at_risk = is_active and max_hp > 0 and hp * 5 <= max_hp * 2
        successor_line = any(
            token in name
            for token in ("Morgrem", "Grimmsnarl", "Riolu", "Lucario", "Makuhita", "Hariyama")
        )
        candidates.append(
            (
                (
                    1 if active_at_risk else 0,
                    0 if (not is_active and successor_line) else 1,
                    1 if is_active else 0,
                    _attached_energy_count(target),
                ),
                index,
            )
        )
    return min(candidates)[1] if candidates else None


def _merge_preference(
    current: tuple[int, ...], candidate: int | None, enabled: bool = True
) -> tuple[int, ...]:
    if current or candidate is None or not enabled:
        return current
    return (candidate,)


def tactical_hard_mask_options(
    profile: str,
    observation: dict[str, Any],
    options: list[dict[str, Any]],
    *,
    features: Any | None = None,
    cards: dict[int, dict[str, Any]],
    attacks: dict[int, dict[str, Any]],
    lucario_aura_hard_mask: bool = False,
    terminal_ko_hard_mask: bool = True,
    forced_deckout_hard_mask: bool = True,
    dragapult_evolve_end_hard_mask: bool = True,
) -> tuple[int, ...]:
    """Return only actions that are provably dominated in the current state.

    The Lucario mask is intentionally narrow: both attacks must be legal, Aura
    Jab must already KO the active, and its discard-to-bench acceleration must
    have real value.  It never masks attacks merely because their damage is
    prevented, and it does not apply to any Dragapult action.
    """

    normalized_profile = profile.strip().lower()
    masked: set[int] = set()

    if lucario_aura_hard_mask and normalized_profile.startswith("lucario"):
        aura = _attack_option_by_name(options, attacks, "Aura Jab")
        mega = _attack_option_by_name(options, attacks, "Mega Brave")
        opponent_hp = _opponent_active_hp(observation)
        if aura is not None and mega is not None and opponent_hp is not None and opponent_hp > 0:
            aura_damage = int(aura[2].get("damage", 0) or 0)
            mega_damage = int(mega[2].get("damage", 0) or 0)
            if (
                aura_damage >= opponent_hp
                and mega_damage >= opponent_hp
                and _lucario_aura_acceleration_available(observation, cards)
            ):
                masked.add(mega[0])

    current = observation.get("current") or {}
    players = current.get("players") or []
    actor = int(current.get("yourIndex", 0) or 0)
    own_player = (
        players[actor]
        if 0 <= actor < len(players) and isinstance(players[actor], dict)
        else {}
    )

    # With five prizes already taken, any unprevented Active KO is a certain
    # game-ending prize. Preserve every lethal attack and remove every action
    # that voluntarily passes that deterministic win.
    if terminal_ko_hard_mask and len(own_player.get("prize") or []) >= 5:
        opponent_hp = _opponent_active_hp(observation)
        prevented = _active_is_ex_wall(observation, cards) and _own_active_is_ex(
            observation, cards
        )
        lethal = {
            index
            for index, option in enumerate(options)
            if int(option.get("type", -1)) == TYPE_ATTACK
            and opponent_hp is not None
            and opponent_hp > 0
            and int(attacks.get(int(option.get("attackId", -1)), {}).get("damage", 0) or 0)
            >= opponent_hp
            and not prevented
        }
        if lethal:
            masked.update(index for index in range(len(options)) if index not in lethal)

    # Explicit competition hotfix: when Dragapult ex is a legal evolution,
    # remove only the option to end the turn without taking any other action.
    if (
        dragapult_evolve_end_hard_mask
        and normalized_profile.startswith("dragapult")
        and features is not None
        and any(
            int(option.get("type", -1)) == TYPE_EVOLVE
            and _option_name(observation, option, features, cards) == "Dragapult ex"
            for option in options
        )
    ):
        masked.update(
            index
            for index, option in enumerate(options)
            if int(option.get("type", -1)) == TYPE_END
        )

    # Mask only explicit fixed-count draw effects that consume the entire
    # remaining deck. Hand-to-deck/bottom recycle effects are deliberately
    # excluded. Keep at least one legal alternative.
    if forced_deckout_hard_mask and features is not None:
        deck_count = int(own_player.get("deckCount", 0) or 0)
        draw_options: set[int] = set()
        if deck_count > 0:
            for index, option in enumerate(options):
                if int(option.get("type", -1)) != TYPE_PLAY:
                    continue
                primary, _ = _option_cards(observation, option, features)
                metadata = cards.get(_card_id(primary), {})
                text = " ".join(
                    str(value.get("text") or "")
                    for value in metadata.get("skills") or []
                    if isinstance(value, dict)
                )
                lowered = text.lower()
                if "shuffle your hand into your deck" in lowered or "bottom of your deck" in lowered:
                    continue
                counts = [int(value) for value in re.findall(r"draw (\d+) cards?", lowered)]
                if counts and max(counts) >= deck_count:
                    draw_options.add(index)
        if draw_options and len(draw_options) < len(options):
            masked.update(draw_options)

    # Never invalidate the entire legal action set.
    return tuple(sorted(masked)) if len(masked) < len(options) else ()


def tactical_search_logit_biases(
    profile: str,
    observation: dict[str, Any],
    options: list[dict[str, Any]],
    *,
    features: Any,
    cards: dict[int, dict[str, Any]],
    attacks: dict[int, dict[str, Any]],
    depth: int = 0,
    scale: float = 1.0,
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """Cheap, auditable 1–2 ply proxies for Dragapult terminal planning."""

    if depth not in (0, 1, 2):
        raise ValueError("tactical search depth must be 0, 1, or 2")
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("tactical search bias scale must be finite and non-negative")
    biases = [0.0] * len(options)
    if depth == 0 or not profile.strip().lower().startswith("dragapult"):
        return tuple(biases), ()

    current = observation.get("current") or {}
    players = current.get("players") or []
    actor = int(current.get("yourIndex", 0) or 0)
    own_player = (
        players[actor]
        if 0 <= actor < len(players) and isinstance(players[actor], dict)
        else {}
    )
    own_board = _own_board(observation)
    opponent_board = _opponent_board(observation)
    opponent_hp = _opponent_active_hp(observation) or 0
    wall_active = _active_is_ex_wall(observation, cards)
    own_ex_active = _own_active_is_ex(observation, cards)
    active_name = _active_name(observation, cards, opponent=False)
    phantom = _attack_option_by_name(options, attacks, "Phantom Dive")
    phantom_attack = phantom[2] if phantom is not None else {"energies": [2, 5]}
    ready_dragapult = any(
        _card_name(row, cards) == "Dragapult ex"
        and _has_attack_energy(row, phantom_attack)
        for row in own_board[1:]
    )
    safe_evolutions: list[int] = []
    two_ply_ready_evolutions: set[int] = set()
    for index, option in enumerate(options):
        if (
            int(option.get("type", -1)) == TYPE_EVOLVE
            and _option_name(observation, option, features, cards) == "Dragapult ex"
        ):
            safe_evolutions.append(index)
            _, target = _option_cards(observation, option, features)
            if depth >= 2 and isinstance(target, dict) and _has_attack_energy(
                target, phantom_attack
            ):
                two_ply_ready_evolutions.add(index)

    opportunities: list[str] = []
    maximum_damage = _maximum_base_attack_damage(options, attacks)
    deck_count = int(own_player.get("deckCount", 0) or 0)
    hand_count = int(own_player.get("handCount", 0) or 0)
    for index, option in enumerate(options):
        option_type = int(option.get("type", -1))
        name = _option_name(observation, option, features, cards)
        if option_type == TYPE_ATTACK:
            attack = attacks.get(int(option.get("attackId", -1)), {})
            attack_name = str(attack.get("name") or "")
            damage = int(attack.get("damage", 0) or 0)
            prevented = bool(wall_active and own_ex_active)
            if opponent_hp > 0 and damage >= opponent_hp and not prevented:
                biases[index] += 1.00
                opportunities.append("dragapult_search_immediate_prize")
            if (
                ready_dragapult
                and active_name in {"Budew", "Munkidori"}
                and attack_name in {"Itchy Pollen", "Mind Bend"}
            ):
                biases[index] -= 1.25
                opportunities.append("dragapult_search_ready_attacker_takeover")
            if attack_name == "Phantom Dive" and wall_active and len(opponent_board) > 1:
                biases[index] += 0.20
                opportunities.append("dragapult_search_phantom_bench_route")
        elif index in safe_evolutions:
            biases[index] += 0.90
            opportunities.append("dragapult_search_safe_evolution")
            if index in two_ply_ready_evolutions:
                biases[index] += 0.40
                opportunities.append("dragapult_search_next_attacker")
        elif option_type == TYPE_END and (safe_evolutions or ready_dragapult):
            biases[index] -= 1.10
            opportunities.append("dragapult_search_avoid_early_end")
        elif option_type == TYPE_PLAY and wall_active and _is_boss_name(name):
            if maximum_damage > 0 and any(
                0 < int(row.get("hp", 0) or 0) <= maximum_damage
                for row in opponent_board[1:]
            ):
                biases[index] += 0.70
                opportunities.append("dragapult_search_boss_prize_route")
        if 0 < deck_count <= 3 and option_type == TYPE_PLAY:
            if name == "Poké Pad":
                biases[index] -= 1.00
                opportunities.append("dragapult_search_preserve_terminal_deck")
            elif name == "Lillie's Determination" and hand_count >= 6:
                biases[index] += 0.35
                opportunities.append("dragapult_search_recycle_terminal_hand")

    return (
        tuple(max(-2.0, min(2.0, value * scale)) for value in biases),
        tuple(dict.fromkeys(opportunities)),
    )


def tactical_adjustment(
    profile: str,
    observation: dict[str, Any],
    options: list[dict[str, Any]],
    action: list[int],
    *,
    features: Any,
    cards: dict[int, dict[str, Any]],
    attacks: dict[int, dict[str, Any]] | None = None,
    state: TacticalShapingState | None = None,
    boss_reservation_penalty: float = 0.0,
    boss_reservation_preference: bool = False,
    boss_post_play_penalty: float = 0.0,
    boss_post_play_preference: bool = False,
    a02_poffin_decline_penalty: float = 0.0,
    a02_poffin_preference: bool = False,
    a02_munkidori_overfill_penalty: float = 0.05,
    a02_bench_budget_preference: bool = False,
    a02_outcome_gated_ordering: bool = False,
    a02_projected_bench_budget: bool = False,
    successor_attach_preference: bool = False,
    a08_terminal_before_evolve_mode: str = "control",
    a08_gated_attack_penalty: float = 0.10,
    a08_maximum_belt_support_penalty: float = 0.0,
    a08_maximum_belt_preference: bool = False,
    a08_second_attacker_reward: float = 0.0,
    a08_recovery_end_penalty: float = 0.0,
    a08_recovery_preference: bool = False,
    end_with_attack_penalty: float = 0.0,
    end_with_attack_preference: bool = False,
    lucario_evolve_penalty: float = 0.0,
    lucario_attach_penalty: float = 0.0,
    lucario_aura_overkill_penalty: float = 0.0,
    lucario_ordering_preference: bool = False,
    dragapult_ready_attacker_penalty: float = 0.14,
    dragapult_evolve_penalty: float = 0.16,
    dragapult_wall_penalty: float = 0.0,
    dragapult_budew_overstay_penalty: float = 0.08,
    dragapult_resource_penalty: float = 0.08,
    dragapult_wall_preference: bool = False,
) -> TacticalAdjustment:
    """Return small, auditable intermediate rewards for known tactical failures.

    Terminal game outcome remains the dominant +/-1 reward.  These adjustments
    only distinguish high-value action ordering decisions observed in public
    replay losses; they intentionally do not penalize every attack with another
    playable card available.
    """

    normalized = profile.strip().lower()
    if normalized in {"", "none", "off"}:
        return TacticalAdjustment(0.0, ())
    chosen = [options[index] for index in action if 0 <= index < len(options)]
    available_types = {int(row.get("type", -1)) for row in options}
    chosen_types = {int(row.get("type", -1)) for row in chosen}
    terminal = bool(chosen_types.intersection({TYPE_ATTACK, TYPE_END}))
    events: list[str] = []
    opportunities: list[str] = []
    reward = 0.0
    preferred_action: tuple[int, ...] = ()
    chose_attack = TYPE_ATTACK in chosen_types
    chose_end = TYPE_END in chosen_types
    maximum_damage = _maximum_base_attack_damage(options, attacks or {})
    first_attack = _first_option_of_type(options, TYPE_ATTACK)

    if (chose_end or chose_attack) and maximum_damage > 0:
        penalty = float(end_with_attack_penalty)
        if not math.isfinite(penalty) or penalty < 0:
            raise ValueError("end-with-attack penalty must be finite and non-negative")
        opportunities.append(f"{normalized}_damaging_attack_before_end")
        if chose_end:
            reward -= penalty
            if penalty > 0:
                events.append(f"{normalized}_ended_with_damaging_attack_available")
            preferred_action = _merge_preference(
                preferred_action, first_attack, end_with_attack_preference
            )

    # Preserve the legacy A08 arm as `control`.  The gated arm defers attack
    # reward and preference shaping until KO, prize, and terminal effects are
    # observable; correct prize-race attacks therefore remain untouched.
    if normalized.startswith("a08"):
        mode = a08_terminal_before_evolve_mode.strip().lower()
        if mode not in A08_TERMINAL_BEFORE_EVOLVE_MODES:
            raise ValueError(f"unknown A08 terminal-before-evolve mode: {mode}")
        gated_penalty = float(a08_gated_attack_penalty)
        if not math.isfinite(gated_penalty) or gated_penalty < 0:
            raise ValueError("A08 gated attack penalty must be finite and non-negative")
        belt_penalty = float(a08_maximum_belt_support_penalty)
        if not math.isfinite(belt_penalty) or belt_penalty < 0:
            raise ValueError(
                "A08 Maximum Belt support penalty must be finite and non-negative"
            )
        evolutions = [
            (index, option)
            for index, option in enumerate(options)
            if int(option.get("type", -1)) == TYPE_EVOLVE
        ]
        high_value_evolutions = [
            (index, option)
            for index, option in evolutions
            if _is_high_value_evolution(observation, option, features, cards)
        ]
        chose_high_value_evolution = any(
            index in action for index, _ in high_value_evolutions
        )
        evolve_target_active = any(
            int(option.get("inPlayArea", -1)) == 4 for _, option in evolutions
        )
        should_penalize_now = bool(evolutions) and (
            (mode == "control" and (chose_attack or chose_end))
            or (mode in {"end_only", "gated"} and chose_end)
        )
        if chose_high_value_evolution:
            opportunities.append("a08_high_value_evolve_before_terminal")
        if should_penalize_now:
            opportunities.append("a08_high_value_evolve_before_terminal")
            reward -= 0.35
            events.append("a08_terminal_before_evolve")
            preferred_action = (evolutions[0][0],)

        board_names = Counter(_card_name(row, cards) for row in _own_board(observation))
        dipplin_count = board_names.get("Dipplin", 0)
        attacker_setup = [
            index
            for index, option in enumerate(options)
            if (
                int(option.get("type", -1)) == TYPE_PLAY
                and _option_name(observation, option, features, cards) == "Applin"
            )
            or (
                int(option.get("type", -1)) == TYPE_EVOLVE
                and _option_name(observation, option, features, cards) == "Dipplin"
            )
        ]
        chose_attacker_setup = any(index in action for index in attacker_setup)
        if attacker_setup and dipplin_count < 2 and (chose_attacker_setup or chose_end):
            opportunity = (
                "a08_board_wipe_recovery"
                if dipplin_count == 0
                else "a08_second_attacker_setup"
            )
            opportunities.append(opportunity)
            setup_reward = float(a08_second_attacker_reward)
            recovery_penalty = float(a08_recovery_end_penalty)
            if not math.isfinite(setup_reward) or setup_reward < 0:
                raise ValueError("A08 second-attacker reward must be finite and non-negative")
            if not math.isfinite(recovery_penalty) or recovery_penalty < 0:
                raise ValueError("A08 recovery penalty must be finite and non-negative")
            if chose_attacker_setup:
                reward += setup_reward
                if setup_reward > 0:
                    events.append("a08_built_next_attacker")
            elif chose_end and not should_penalize_now:
                reward -= recovery_penalty
                if recovery_penalty > 0:
                    events.append(
                        "a08_ended_before_board_wipe_recovery"
                        if dipplin_count == 0
                        else "a08_ended_before_second_attacker_setup"
                    )
                preferred_action = _merge_preference(
                    preferred_action,
                    attacker_setup[0],
                    a08_recovery_preference,
                )
        belt_targets: list[tuple[int, str]] = []
        for index, option in enumerate(options):
            if int(option.get("type", -1)) != TYPE_ATTACH:
                continue
            primary, target = _option_cards(observation, option, features)
            if _card_name(primary, cards) != "Maximum Belt":
                continue
            belt_targets.append((index, _card_name(target, cards)))
        attacker_belt_targets = [
            (index, target_name)
            for index, target_name in belt_targets
            if target_name in A08_MAXIMUM_BELT_ATTACKERS
        ]
        chose_support_belt = any(
            index in action and target_name in A08_MAXIMUM_BELT_SUPPORTS
            for index, target_name in belt_targets
        )
        if chose_support_belt and attacker_belt_targets:
            reward -= belt_penalty
            events.append("a08_maximum_belt_on_support_with_attacker_available")
            if a08_maximum_belt_preference and not preferred_action:
                # Prefer the already evolved attacker, then a future attacker.
                attacker_belt_targets.sort(
                    key=lambda row: (
                        0 if row[1] == "Dipplin" else 1,
                        row[0],
                    )
                )
                preferred_action = (attacker_belt_targets[0][0],)
        if mode == "gated" and high_value_evolutions and (
            chose_attack or chose_high_value_evolution
        ):
            opportunities.append("a08_high_value_evolve_before_nonlethal_attack")
        if mode == "gated" and chose_attack and high_value_evolutions:
            return TacticalAdjustment(
                reward,
                tuple(events),
                preferred_action,
                deferred_attack_penalty=gated_penalty,
                deferred_preferred_action=(high_value_evolutions[0][0],),
                evolve_target_active=evolve_target_active,
                deferred_attack_events=(
                    "a08_nonlethal_attack_before_high_value_evolve",
                ),
                opportunities=tuple(dict.fromkeys(opportunities)),
            )
        return TacticalAdjustment(
            reward,
            tuple(events),
            preferred_action,
            evolve_target_active=evolve_target_active,
            opportunities=tuple(dict.fromkeys(opportunities)),
        )

    if normalized.startswith("dragapult"):
        attacks = attacks or {}
        for value, label in (
            (dragapult_ready_attacker_penalty, "Dragapult ready-attacker"),
            (dragapult_evolve_penalty, "Dragapult evolution"),
            (dragapult_wall_penalty, "Dragapult wall"),
            (dragapult_budew_overstay_penalty, "Dragapult Budew overstay"),
            (dragapult_resource_penalty, "Dragapult wall resource"),
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{label} penalty must be finite and non-negative")

        current = observation.get("current") or {}
        players = current.get("players") or []
        actor = int(current.get("yourIndex", 0) or 0)
        own_player = (
            players[actor]
            if 0 <= actor < len(players) and isinstance(players[actor], dict)
            else {}
        )
        own_board = _own_board(observation)
        opponent_board = _opponent_board(observation)
        own_names = Counter(_card_name(row, cards) for row in own_board)
        bench_count = max(len(own_board) - 1, 0)
        deck_count = int(own_player.get("deckCount", 0) or 0)
        hand_count = int(own_player.get("handCount", 0) or 0)
        wall_active = _active_is_ex_wall(observation, cards)
        phantom = _attack_option_by_name(options, attacks, "Phantom Dive")
        itchy = _attack_option_by_name(options, attacks, "Itchy Pollen")
        mind_bend = _attack_option_by_name(options, attacks, "Mind Bend")
        phantom_attack = phantom[2] if phantom is not None else {
            "energies": [2, 5]
        }
        ready_dragapult = [
            row
            for row in own_board[1:]
            if _card_name(row, cards) == "Dragapult ex"
            and _has_attack_energy(row, phantom_attack)
        ]
        safe_dragapult_evolutions = [
            index
            for index, option in enumerate(options)
            if int(option.get("type", -1)) == TYPE_EVOLVE
            and _option_name(observation, option, features, cards) == "Dragapult ex"
        ]
        boss_options = [
            index
            for index, option in enumerate(options)
            if int(option.get("type", -1)) == TYPE_PLAY
            and _is_boss_name(_option_name(observation, option, features, cards))
        ]
        chosen_names = Counter(
            _option_name(observation, option, features, cards) for option in chosen
        )
        chosen_other_supporter = any(
            int(option.get("type", -1)) == TYPE_PLAY
            and not _is_boss_name(_option_name(observation, option, features, cards))
            and _is_supporter(_option_cards(observation, option, features)[0], cards)
            for option in chosen
        )
        deferred_penalty = 0.0
        deferred_events: list[str] = []
        deferred_preference: tuple[int, ...] = ()

        # Phantom Dive is never masked or penalized against an ex wall: its
        # bench counters can still map prizes.  Record the state for audit only.
        if wall_active and phantom is not None:
            opportunities.append("dragapult_phantom_full_value_search")

        active_name = _active_name(observation, cards, opponent=False)
        chose_stall_attack = bool(
            (itchy is not None and itchy[0] in action)
            or (mind_bend is not None and mind_bend[0] in action)
        )

        # When a fully powered Dragapult is already on the bench, another
        # Budew/Munkidori stall attack is the replay-proven launch failure.
        # Keep this outcome-gated so a surprise KO/prize cancels the penalty.
        ready_takeover = bool(
            ready_dragapult and active_name in {"Budew", "Munkidori"}
        )
        if ready_takeover:
            opportunities.append("dragapult_ready_attacker_takeover")
            if chose_stall_attack:
                deferred_penalty += float(dragapult_ready_attacker_penalty)
                deferred_events.append(
                    "dragapult_delayed_ready_attacker_with_stall_action"
                )
            elif chose_end:
                reward -= float(dragapult_ready_attacker_penalty)
                if float(dragapult_ready_attacker_penalty) > 0:
                    events.append("dragapult_delayed_ready_attacker_with_stall_action")

        # Two legal Dragapult evolutions followed by END is immediately wrong.
        # A stall attack is gated on its result, since item lock can occasionally
        # be correct; a KO/prize cancels this shaping signal.
        if safe_dragapult_evolutions and (chose_end or chose_stall_attack):
            opportunities.append("dragapult_safe_evolution_before_terminal")
            if chose_end:
                reward -= float(dragapult_evolve_penalty)
                if float(dragapult_evolve_penalty) > 0:
                    events.append(
                        "dragapult_terminal_before_safe_dragapult_evolution"
                    )
                preferred_action = _merge_preference(
                    preferred_action, safe_dragapult_evolutions[0], True
                )
            else:
                deferred_penalty += float(dragapult_evolve_penalty)
                deferred_events.append(
                    "dragapult_nonprize_stall_before_safe_dragapult_evolution"
                )
                deferred_preference = (safe_dragapult_evolutions[0],)

        # Boss is the cleanest wall bypass.  Penalize spending the once-per-turn
        # Supporter on something else only when Boss is already legal and there
        # is a real benched target; do not guess about a Boss still in the deck.
        wall_boss_opportunity = bool(
            wall_active
            and boss_options
            and maximum_damage > 0
            and any(
                0 < int(row.get("hp", 0) or 0) <= maximum_damage
                for row in opponent_board[1:]
            )
        )
        if wall_boss_opportunity:
            opportunities.append("dragapult_ex_wall_bypass")
            if chosen_other_supporter:
                reward -= float(dragapult_wall_penalty)
                if float(dragapult_wall_penalty) > 0:
                    events.append("dragapult_consumed_supporter_before_wall_boss")
                preferred_action = _merge_preference(
                    preferred_action,
                    boss_options[0],
                    dragapult_wall_preference,
                )

        # Going second may use Budew twice to slow setup, but once a Drakloak or
        # Dragapult attack line exists, repeated Itchy Pollen becomes a launch
        # failure.  Going first gets one additional grace use.  A surprise KO
        # still cancels the deferred penalty.
        if state is not None and itchy is not None and active_name == "Budew":
            key = (actor, "Itchy Pollen")
            prior_uses = int(state.attack_counts.get(key, 0))
            first_player = int(current.get("firstPlayer", -1) or 0)
            going_second = first_player in (0, 1) and actor != first_player
            grace_uses = 2 if going_second else 3
            attack_line_ready = bool(
                own_names.get("Drakloak", 0) or own_names.get("Dragapult ex", 0)
            )
            if attack_line_ready and prior_uses >= grace_uses and not ready_takeover:
                opportunities.append("dragapult_budew_launch_budget")
                if itchy[0] in action:
                    deferred_penalty += float(dragapult_budew_overstay_penalty)
                    deferred_events.append(
                        "dragapult_overstayed_budew_after_attack_line_ready"
                    )
                    transition = next(
                        (
                            index
                            for index, option in enumerate(options)
                            if int(option.get("type", -1))
                            in (TYPE_EVOLVE, TYPE_ATTACH)
                        ),
                        None,
                    )
                    if transition is not None and not deferred_preference:
                        deferred_preference = (transition,)
            if itchy[0] in action:
                state.attack_counts[key] += 1

        # A full bench is not inherently wrong.  Only discourage adding the two
        # low-value fillers observed in wall losses when four slots are already
        # occupied; Munkidori and the Dragapult evolution line remain exempt.
        low_value_fillers = {"Budew", "Meowth ex"}
        filler_options = [
            index
            for index, option in enumerate(options)
            if int(option.get("type", -1)) == TYPE_PLAY
            and _option_name(observation, option, features, cards)
            in low_value_fillers
        ]
        if wall_active and bench_count >= 4 and filler_options:
            opportunities.append("dragapult_wall_resource_budget")
            if any(index in action for index in filler_options):
                reward -= float(dragapult_resource_penalty)
                if float(dragapult_resource_penalty) > 0:
                    events.append("dragapult_played_low_value_bench_filler_into_wall")
                if dragapult_wall_preference and boss_options:
                    preferred_action = _merge_preference(
                        preferred_action, boss_options[0]
                    )

        # Do not blanket-penalize Lillie's Determination: it can put a large hand
        # back.  Poké Pad always spends scarce deck budget, so at deck<=3 it is
        # a narrow terminal-resource error in every matchup, not only walls.
        poke_pad_options = [
            index
            for index, option in enumerate(options)
            if int(option.get("type", -1)) == TYPE_PLAY
            and _option_name(observation, option, features, cards) == "Poké Pad"
        ]
        if (
            0 < deck_count <= 3
            and hand_count >= 6
            and poke_pad_options
        ):
            opportunities.append("dragapult_wall_resource_budget")
            if any(index in action for index in poke_pad_options):
                reward -= float(dragapult_resource_penalty)
                if float(dragapult_resource_penalty) > 0:
                    events.append("dragapult_searched_thin_deck_into_wall")
                if dragapult_wall_preference and wall_active and boss_options:
                    preferred_action = _merge_preference(
                        preferred_action, boss_options[0]
                    )

        return TacticalAdjustment(
            reward,
            tuple(events),
            preferred_action,
            deferred_attack_penalty=min(deferred_penalty, 0.30),
            deferred_preferred_action=deferred_preference,
            deferred_attack_events=tuple(deferred_events),
            opportunities=tuple(dict.fromkeys(opportunities)),
        )

    if normalized.startswith("lucario"):
        attacks = attacks or {}
        aura = _attack_option_by_name(options, attacks, "Aura Jab")
        mega = _attack_option_by_name(options, attacks, "Mega Brave")
        opponent_hp = _opponent_active_hp(observation)
        aura_damage = int((aura or ({}, {}, {}))[2].get("damage", 0) or 0)
        aura_lethal_with_acceleration = (
            aura is not None
            and mega is not None
            and opponent_hp is not None
            and opponent_hp > 0
            and aura_damage >= opponent_hp
            and _lucario_aura_acceleration_available(observation, cards)
        )
        if aura_lethal_with_acceleration:
            opportunities.append("lucario_aura_jab_lethal_acceleration")
            overkill_penalty = float(lucario_aura_overkill_penalty)
            if not math.isfinite(overkill_penalty) or overkill_penalty < 0:
                raise ValueError(
                    "Lucario Aura-overkill penalty must be finite and non-negative"
                )
            chose_mega = mega[0] in action
            if chose_mega:
                reward -= overkill_penalty
                if overkill_penalty > 0:
                    events.append(
                        "lucario_mega_brave_overkill_forfeited_aura_acceleration"
                    )
                preferred_action = _merge_preference(
                    preferred_action,
                    aura[0],
                    lucario_ordering_preference,
                )
        evolve_options = [
            (index, option)
            for index, option in enumerate(options)
            if int(option.get("type", -1)) == TYPE_EVOLVE
        ]
        high_value = [
            (index, option)
            for index, option in evolve_options
            if "Mega Lucario ex" in _option_name(observation, option, features, cards)
            or "Hariyama" in _option_name(observation, option, features, cards)
            or _is_high_value_evolution(observation, option, features, cards)
        ]
        chose_high_value = any(index in action for index, _ in high_value)
        attach_option = _successor_attach_option(observation, options, features, cards)
        can_attach = (
            attach_option is not None
            and not bool((observation.get("current") or {}).get("energyAttached"))
        )
        evolve_penalty = float(lucario_evolve_penalty)
        attach_penalty = float(lucario_attach_penalty)
        for value, label in (
            (evolve_penalty, "Lucario evolve"),
            (attach_penalty, "Lucario attach"),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} penalty must be finite and non-negative")

        deferred_penalty = 0.0
        deferred_events: list[str] = []
        deferred_preference: tuple[int, ...] = ()
        if high_value and chose_high_value:
            opportunities.append("lucario_high_value_evolve_before_terminal")
        if high_value and (chose_attack or chose_end):
            opportunities.append("lucario_high_value_evolve_before_terminal")
            if chose_end:
                reward -= evolve_penalty
                if evolve_penalty > 0:
                    events.append("lucario_end_before_high_value_evolve")
                preferred_action = _merge_preference(
                    preferred_action, high_value[0][0], lucario_ordering_preference
                )
            else:
                deferred_penalty += evolve_penalty
                deferred_events.append(
                    "lucario_nonlethal_attack_before_high_value_evolve"
                )
                deferred_preference = (high_value[0][0],)
        chose_attach = TYPE_ATTACH in chosen_types
        if can_attach and chose_attach:
            opportunities.append("lucario_attach_before_terminal")
        if can_attach and (chose_attack or chose_end):
            opportunities.append("lucario_attach_before_terminal")
            if chose_end:
                reward -= attach_penalty
                if attach_penalty > 0:
                    events.append("lucario_end_before_manual_attach")
                preferred_action = _merge_preference(
                    preferred_action,
                    attach_option,
                    lucario_ordering_preference or successor_attach_preference,
                )
            else:
                deferred_penalty += attach_penalty
                deferred_events.append("lucario_nonlethal_attack_before_manual_attach")
                if not deferred_preference and attach_option is not None:
                    deferred_preference = (attach_option,)
        return TacticalAdjustment(
            reward,
            tuple(events),
            preferred_action,
            deferred_attack_penalty=min(deferred_penalty, 0.30),
            deferred_preferred_action=deferred_preference,
            deferred_attack_events=tuple(deferred_events),
            opportunities=tuple(dict.fromkeys(opportunities)),
        )

    if not normalized.startswith("a02"):
        raise ValueError(f"unknown tactical shaping profile: {profile}")

    board = _own_board(observation)
    board_names = Counter(_card_name(row, cards) for row in board)
    bench_count = max(len(board) - 1, 0)
    open_bench = max(5 - bench_count, 0)

    grim_evolutions = []
    setup_plays: set[str] = set()
    boss_available = False
    for option in options:
        option_type = int(option.get("type", -1))
        name = _option_name(observation, option, features, cards)
        if option_type == TYPE_EVOLVE and "Grimmsnarl ex" in name:
            grim_evolutions.append(option)
        if option_type == TYPE_PLAY and name in {"Snorunt", "Marnie's Impidimp"}:
            setup_plays.add(name)
        if option_type == TYPE_PLAY and _is_boss_name(name):
            boss_available = True

    grim_count = sum(count for name, count in board_names.items() if "Grimmsnarl ex" in name)
    froslass_count = board_names.get("Froslass", 0)
    current = observation.get("current") or {}
    players = current.get("players") or []
    actor = int(current.get("yourIndex", 0) or 0)
    own_player = (
        players[actor]
        if 0 <= actor < len(players) and isinstance(players[actor], dict)
        else {}
    )
    hand_names = Counter(
        _card_name(row, cards)
        for row in (own_player.get("hand") or [])
        if isinstance(row, dict)
    )
    deferred_penalty = 0.0
    deferred_events: list[str] = []
    deferred_preferred_action: tuple[int, ...] = ()
    chosen_names = Counter(_option_name(observation, row, features, cards) for row in chosen)
    chosen_play_names = Counter(
        _option_name(observation, row, features, cards)
        for row in chosen
        if int(row.get("type", -1)) == TYPE_PLAY
    )

    def record_terminal_ordering(
        *,
        penalty: float,
        event: str,
        opportunity: str,
        preferred: int | None,
    ) -> None:
        nonlocal reward, preferred_action, deferred_penalty
        nonlocal deferred_preferred_action
        opportunities.append(opportunity)
        if chose_attack and a02_outcome_gated_ordering:
            deferred_penalty += penalty
            deferred_events.append(event.replace("terminal", "nonlethal_attack"))
            if not deferred_preferred_action and preferred is not None:
                deferred_preferred_action = (preferred,)
            return
        reward -= penalty
        events.append(event)
        preferred_action = _merge_preference(preferred_action, preferred)

    # Poffin is a multi-decision effect.  Shape the actual optional-selection
    # row instead of the earlier PLAY row so the preference points at Snorunt.
    select = observation.get("select") or {}
    effect_name = _card_name(select.get("effect"), cards)
    snorunt_choices = [
        index
        for index, option in enumerate(options)
        if _option_name(observation, option, features, cards) == "Snorunt"
    ]
    chose_snorunt = any(index in action for index in snorunt_choices)
    poffin_setup_opportunity = (
        effect_name == "Buddy-Buddy Poffin"
        and bool(snorunt_choices)
        and open_bench > 0
        and froslass_count == 0
        and hand_names.get("Froslass", 0) > 0
    )
    if poffin_setup_opportunity:
        opportunities.append("a02_poffin_snorunt_setup")
    if poffin_setup_opportunity and not chose_snorunt:
        poffin_penalty = float(a02_poffin_decline_penalty)
        if not math.isfinite(poffin_penalty) or poffin_penalty < 0:
            raise ValueError("A02 Poffin decline penalty must be finite and non-negative")
        reward -= poffin_penalty
        events.append("a02_declined_snorunt_after_poffin_with_froslass_in_hand")
        if a02_poffin_preference and not preferred_action:
            preferred_action = (snorunt_choices[0],)
    chose_grim_evolution = any(option in chosen for option in grim_evolutions)
    if chose_grim_evolution and grim_count < 2:
        opportunities.append("a02_second_grimmsnarl_before_terminal")
    if terminal and grim_evolutions and grim_count < 2:
        record_terminal_ordering(
            penalty=0.30,
            event="a02_terminal_before_second_grimmsnarl",
            opportunity="a02_second_grimmsnarl_before_terminal",
            preferred=options.index(grim_evolutions[0]),
        )
    chose_snorunt_setup = chosen_play_names.get("Snorunt", 0) > 0
    if chose_snorunt_setup and open_bench > 0 and froslass_count == 0:
        opportunities.append("a02_froslass_setup_before_terminal")
    if terminal and open_bench > 0 and "Snorunt" in setup_plays and froslass_count == 0:
        record_terminal_ordering(
            penalty=0.12,
            event="a02_terminal_before_froslass_setup",
            opportunity="a02_froslass_setup_before_terminal",
            preferred=next(
                index for index, option in enumerate(options)
                if int(option.get("type", -1)) == TYPE_PLAY
                and _option_name(observation, option, features, cards) == "Snorunt"
            ),
        )
    chose_grim_setup = chosen_play_names.get("Marnie's Impidimp", 0) > 0
    if chose_grim_setup and open_bench > 0 and grim_count < 2:
        opportunities.append("a02_grimmsnarl_setup_before_terminal")
    if terminal and open_bench > 0 and "Marnie's Impidimp" in setup_plays and grim_count < 2:
        record_terminal_ordering(
            penalty=0.10,
            event="a02_terminal_before_grimmsnarl_setup",
            opportunity="a02_grimmsnarl_setup_before_terminal",
            preferred=next(
                index for index, option in enumerate(options)
                if int(option.get("type", -1)) == TYPE_PLAY
                and _option_name(observation, option, features, cards) == "Marnie's Impidimp"
            ),
        )

    attach_opportunity = (
        TYPE_ATTACH in available_types and not bool(current.get("energyAttached"))
    )
    if attach_opportunity and TYPE_ATTACH in chosen_types:
        opportunities.append("a02_manual_attach_before_terminal")
    if terminal and attach_opportunity:
        attach_preference = (
            _successor_attach_option(observation, options, features, cards)
            if successor_attach_preference
            else _first_option_of_type(options, TYPE_ATTACH)
        )
        record_terminal_ordering(
            penalty=0.06,
            event="a02_terminal_before_manual_attach",
            opportunity="a02_manual_attach_before_terminal",
            preferred=attach_preference,
        )

    # Boss prize mapping trigger: preserve it as a narrow hard-negative signal
    # only when a low-HP opposing bench target exists and the active is durable.
    opponent_board = _opponent_board(observation)
    active_hp = int(opponent_board[0].get("hp", 0) or 0) if opponent_board else 0
    bench_hps = [int(row.get("hp", 0) or 0) for row in opponent_board[1:]]
    boss_mapping_opportunity = (
        boss_available and active_hp >= 150 and any(0 < hp <= 90 for hp in bench_hps)
    )
    chosen_boss_row = any(
        int(row.get("type", -1)) == TYPE_PLAY
        and _is_boss_name(_option_name(observation, row, features, cards))
        for row in chosen
    )
    if boss_mapping_opportunity and chosen_boss_row:
        opportunities.append("a02_boss_low_hp_bench_before_terminal")
    if terminal and boss_mapping_opportunity:
        record_terminal_ordering(
            penalty=0.12,
            event="a02_terminal_before_boss_low_hp_bench",
            opportunity="a02_boss_low_hp_bench_before_terminal",
            preferred=next(
                index for index, option in enumerate(options)
                if int(option.get("type", -1)) == TYPE_PLAY
                and _is_boss_name(_option_name(observation, option, features, cards))
            ),
        )

    # Turn-level Boss reservation for the ordering gap that the row-local
    # terminal signal above cannot see: Boss is legal and maps a prize, but a
    # different Supporter is played first and consumes the one-Supporter turn.
    # The base-damage test is deliberately conservative; dynamic bonuses and
    # Weakness are not guessed.  State is optional to preserve existing callers.
    if state is not None:
        turn = int(current.get("turn", 0) or 0)
        if state.boss_reservation_turns.get(actor) not in (None, turn):
            state.boss_reservation_turns.pop(actor, None)
        if state.boss_post_play_turns.get(actor) not in (None, turn):
            state.boss_post_play_turns.pop(actor, None)
        boss_prize_available = (
            boss_available
            and maximum_damage > 0
            and active_hp > maximum_damage
            and any(0 < hp <= min(90, maximum_damage) for hp in bench_hps)
        )
        if boss_prize_available:
            state.boss_reservation_turns[actor] = turn

        chosen_boss = any(
            int(row.get("type", -1)) == TYPE_PLAY
            and _is_boss_name(_option_name(observation, row, features, cards))
            for row in chosen
        )
        chosen_other_supporter = any(
            int(row.get("type", -1)) == TYPE_PLAY
            and not _is_boss_name(_option_name(observation, row, features, cards))
            and _is_supporter(_option_cards(observation, row, features)[0], cards)
            for row in chosen
        )
        reservation_active = state.boss_reservation_turns.get(actor) == turn
        if reservation_active and (chosen_boss or chosen_other_supporter):
            opportunities.append("a02_boss_reserved_before_other_supporter")
        if reservation_active and chosen_other_supporter:
            reward -= max(float(boss_reservation_penalty), 0.0)
            events.append("consumed_supporter_before_boss_low_hp_bench")
            if boss_reservation_preference and not preferred_action:
                preferred_action = (next(
                    index for index, option in enumerate(options)
                    if int(option.get("type", -1)) == TYPE_PLAY
                    and _is_boss_name(_option_name(observation, option, features, cards))
                ),)
        post_play_active = state.boss_post_play_turns.get(actor) == turn
        if post_play_active and (chose_attack or chose_end):
            opportunities.append("a02_attack_follow_through_after_boss")
            post_penalty = float(boss_post_play_penalty)
            if not math.isfinite(post_penalty) or post_penalty < 0:
                raise ValueError("Boss post-play penalty must be finite and non-negative")
            if chose_end and maximum_damage > 0:
                reward -= post_penalty
                if post_penalty > 0:
                    events.append("a02_ended_after_boss_with_attack_available")
                preferred_action = _merge_preference(
                    preferred_action, first_attack, boss_post_play_preference
                )
            state.boss_post_play_turns.pop(actor, None)
        if chosen_boss and reservation_active:
            state.boss_post_play_turns[actor] = turn
        if chosen_other_supporter or terminal:
            state.boss_reservation_turns.pop(actor, None)

    existing_munkidori = board_names.get("Munkidori", 0)
    missing_engine_slots = int(froslass_count == 0) + int(grim_count < 2)
    projected_open_bench = max(open_bench - 1, 0)
    projected_budget_violation = (
        a02_projected_bench_budget
        and existing_munkidori >= 2
        and projected_open_bench < missing_engine_slots
    )
    bench_budget_risk = (
        existing_munkidori >= 3
        or (open_bench <= 1 and (froslass_count == 0 or grim_count < 2))
        or projected_budget_violation
    )
    munkidori_play_available = any(
        int(option.get("type", -1)) == TYPE_PLAY
        and _option_name(observation, option, features, cards) == "Munkidori"
        for option in options
    )
    if munkidori_play_available and bench_budget_risk:
        opportunities.append("a02_projected_bench_budget")
    if chosen_play_names.get("Munkidori", 0) and bench_budget_risk:
        bench_penalty = float(a02_munkidori_overfill_penalty)
        if not math.isfinite(bench_penalty) or bench_penalty < 0:
            raise ValueError(
                "A02 Munkidori overfill penalty must be finite and non-negative"
            )
        reward -= bench_penalty
        events.append("a02_overfilled_munkidori_bench")
        if a02_bench_budget_preference and not preferred_action:
            setup_alternatives = [
                index
                for index, option in enumerate(options)
                if (
                    int(option.get("type", -1)) == TYPE_EVOLVE
                    and (
                        "Grimmsnarl ex"
                        in _option_name(observation, option, features, cards)
                        or _option_name(observation, option, features, cards)
                        == "Froslass"
                    )
                )
                or (
                    int(option.get("type", -1)) == TYPE_PLAY
                    and _option_name(observation, option, features, cards)
                    in {"Snorunt", "Marnie's Impidimp"}
                )
            ]
            if setup_alternatives:
                preferred_action = (setup_alternatives[0],)

    # Give a small symmetric positive signal to the intended setup actions so
    # rare good decisions are not learned only through absence of a penalty.
    if any(
        int(row.get("type", -1)) == TYPE_EVOLVE
        and "Grimmsnarl ex" in _option_name(observation, row, features, cards)
        for row in chosen
    ) and grim_count < 2:
        reward += 0.08
        events.append("a02_completed_second_grimmsnarl")
    if chosen_names.get("Snorunt", 0) and froslass_count == 0:
        reward += 0.04
        events.append("a02_started_froslass_setup")

    return TacticalAdjustment(
        reward,
        tuple(events),
        preferred_action,
        deferred_attack_penalty=min(deferred_penalty, 0.35),
        deferred_preferred_action=deferred_preferred_action,
        deferred_attack_events=tuple(dict.fromkeys(deferred_events)),
        opportunities=tuple(dict.fromkeys(opportunities)),
    )
