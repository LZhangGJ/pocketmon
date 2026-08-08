from __future__ import annotations

import json
import math
import zlib
from pathlib import Path
from typing import Any

import numpy as np


STATE_SCALAR_DIM = 64
STATE_HASH_DIM = 256
OPTION_SCALAR_DIM = 48
OPTION_HASH_DIM = 128
STATE_DIM = STATE_SCALAR_DIM + STATE_HASH_DIM
OPTION_DIM = OPTION_SCALAR_DIM + OPTION_HASH_DIM


def load_catalog(path: str | Path) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cards = {int(card["cardId"]): card for card in payload["cards"]}
    attacks = {int(attack["attackId"]): attack for attack in payload["attacks"]}
    return cards, attacks


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _hash_add(vector: np.ndarray, offset: int, size: int, token: str, value: float = 1.0) -> None:
    code = zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF
    index = code % size
    sign = -1.0 if (code & 0x80000000) else 1.0
    vector[offset + index] += np.float32(sign * value)


def _card_id(card: Any) -> int:
    if isinstance(card, dict) and isinstance(card.get("id"), int):
        return int(card["id"])
    return 0


def _zone_cards(current: dict[str, Any], side: int, zone: str) -> list[Any]:
    players = current.get("players") or []
    if not (0 <= side < len(players)):
        return []
    value = players[side].get(zone)
    return value if isinstance(value, list) else []


def _resolve_area_card(
    observation: dict[str, Any], area: Any, index: Any, player_index: Any
) -> dict[str, Any] | None:
    if not isinstance(index, int) or index < 0:
        return None
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    your_index = int(current.get("yourIndex", 0))
    owner = int(player_index) if isinstance(player_index, int) else your_index

    if area == 1:  # hidden deck cards exposed only during a deck-search selection
        cards = select.get("deck") or []
    elif area == 2:
        cards = _zone_cards(current, owner, "hand")
    elif area == 3:
        cards = _zone_cards(current, owner, "discard")
    elif area == 4:
        cards = _zone_cards(current, owner, "active")
    elif area == 5:
        cards = _zone_cards(current, owner, "bench")
    elif area == 6:
        cards = _zone_cards(current, owner, "prize")
    elif area == 7:
        cards = current.get("stadium") or []
    elif area == 12:
        cards = current.get("looking") or []
    else:
        cards = []
    if index >= len(cards) or not isinstance(cards[index], dict):
        return None
    return cards[index]


def resolve_option_cards(
    observation: dict[str, Any], option: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    option_type = option.get("type")
    current = observation.get("current") or {}
    your_index = int(current.get("yourIndex", 0))
    primary: dict[str, Any] | None = None
    target: dict[str, Any] | None = None

    if option_type == 7:  # PLAY from hand
        primary = _resolve_area_card(observation, 2, option.get("index"), your_index)
    elif option_type in (8, 9):  # ATTACH or EVOLVE
        primary = _resolve_area_card(
            observation, option.get("area"), option.get("index"), your_index
        )
        target = _resolve_area_card(
            observation,
            option.get("inPlayArea"),
            option.get("inPlayIndex"),
            your_index,
        )
    elif option_type in (10, 11):  # ABILITY or DISCARD in play
        primary = _resolve_area_card(
            observation, option.get("area"), option.get("index"), your_index
        )
    elif option_type in (3, 4, 5, 6):
        target = _resolve_area_card(
            observation,
            option.get("area"),
            option.get("index"),
            option.get("playerIndex"),
        )
        primary = target
        if option_type == 4 and target is not None:
            cards = target.get("tools") or []
            i = option.get("toolIndex")
            if isinstance(i, int) and 0 <= i < len(cards) and isinstance(cards[i], dict):
                primary = cards[i]
        elif option_type in (5, 6) and target is not None:
            cards = target.get("energyCards") or []
            i = option.get("energyIndex")
            if isinstance(i, int) and 0 <= i < len(cards) and isinstance(cards[i], dict):
                primary = cards[i]
    elif option_type == 15 and isinstance(option.get("cardId"), int):
        primary = {"id": option["cardId"]}
    return primary, target


def _add_card_tokens(vector: np.ndarray, side_name: str, zone: str, card: Any) -> None:
    card_id = _card_id(card)
    if card_id <= 0:
        return
    base = STATE_SCALAR_DIM
    _hash_add(vector, base, STATE_HASH_DIM, f"card:{card_id}")
    _hash_add(vector, base, STATE_HASH_DIM, f"{side_name}:{zone}:{card_id}")
    if isinstance(card, dict) and isinstance(card.get("hp"), (int, float)):
        max_hp = max(1.0, _number(card.get("maxHp"), 1.0))
        hp_bucket = int(10.0 * max(0.0, min(1.0, _number(card.get("hp")) / max_hp)))
        _hash_add(vector, base, STATE_HASH_DIM, f"{side_name}:{zone}:hp:{hp_bucket}:{card_id}")
        for energy in card.get("energies") or []:
            _hash_add(vector, base, STATE_HASH_DIM, f"{side_name}:{zone}:energy:{energy}:{card_id}")
        for attached_zone in ("energyCards", "tools", "preEvolution"):
            for attached in card.get(attached_zone) or []:
                attached_id = _card_id(attached)
                if attached_id:
                    _hash_add(
                        vector,
                        base,
                        STATE_HASH_DIM,
                        f"{side_name}:{zone}:{attached_zone}:{attached_id}",
                    )


def encode_state(observation: dict[str, Any]) -> np.ndarray:
    vector = np.zeros(STATE_DIM, dtype=np.float32)
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    players = current.get("players") or []
    your_index = int(current.get("yourIndex", 0))
    opponent_index = 1 - your_index
    first_player = int(current.get("firstPlayer", -1))

    vector[0] = 1.0
    vector[1] = np.float32(min(_number(current.get("turn")) / 100.0, 3.0))
    vector[2] = np.float32(min(_number(current.get("turnActionCount")) / 30.0, 3.0))
    vector[3] = float(first_player == your_index)
    vector[4] = float(first_player == opponent_index)
    vector[5] = _number(current.get("supporterPlayed"))
    vector[6] = _number(current.get("stadiumPlayed"))
    vector[7] = _number(current.get("energyAttached"))
    vector[8] = _number(current.get("retreated"))
    vector[9] = min(len(current.get("stadium") or []), 1)
    vector[10] = min(len(current.get("looking") or []) / 20.0, 2.0)
    vector[11] = min(len(observation.get("logs") or []) / 20.0, 3.0)
    vector[12] = _number(select.get("minCount")) / 16.0
    vector[13] = _number(select.get("maxCount")) / 16.0
    vector[14] = len(select.get("option") or []) / 40.0
    vector[15] = _number(select.get("remainEnergyCost")) / 10.0

    for relative_side, player_index in enumerate((your_index, opponent_index)):
        if not (0 <= player_index < len(players)):
            continue
        player = players[player_index]
        active = (player.get("active") or [None])[0] if player.get("active") else None
        bench = player.get("bench") or []
        base = 16 + relative_side * 24
        vector[base + 0] = _number(player.get("deckCount")) / 60.0
        vector[base + 1] = _number(player.get("handCount")) / 20.0
        vector[base + 2] = len(player.get("prize") or []) / 6.0
        vector[base + 3] = len(bench) / 5.0
        vector[base + 4] = _number(player.get("benchMax"), 5.0) / 5.0
        vector[base + 5] = float(isinstance(active, dict))
        if isinstance(active, dict):
            max_hp = max(1.0, _number(active.get("maxHp"), 1.0))
            hp = _number(active.get("hp"))
            vector[base + 6] = hp / 400.0
            vector[base + 7] = (max_hp - hp) / 400.0
            vector[base + 8] = len(active.get("energies") or []) / 10.0
            vector[base + 9] = len(active.get("tools") or []) / 3.0
            vector[base + 10] = len(active.get("preEvolution") or []) / 3.0
            vector[base + 11] = _number(active.get("appearThisTurn"))
        bench_hp = sum(_number(card.get("hp")) for card in bench if isinstance(card, dict))
        bench_max_hp = sum(_number(card.get("maxHp")) for card in bench if isinstance(card, dict))
        vector[base + 12] = bench_hp / 2000.0
        vector[base + 13] = (bench_max_hp - bench_hp) / 2000.0
        vector[base + 14] = sum(
            len(card.get("energies") or []) for card in bench if isinstance(card, dict)
        ) / 30.0
        vector[base + 15] = _number(player.get("poisoned"))
        vector[base + 16] = _number(player.get("burned"))
        vector[base + 17] = _number(player.get("asleep"))
        vector[base + 18] = _number(player.get("paralyzed"))
        vector[base + 19] = _number(player.get("confused"))
        vector[base + 20] = len(player.get("discard") or []) / 60.0
        vector[base + 21] = len(player.get("hand") or []) / 20.0
        vector[base + 22] = sum(
            len(card.get("tools") or []) for card in bench if isinstance(card, dict)
        ) / 10.0
        vector[base + 23] = sum(
            len(card.get("preEvolution") or []) for card in bench if isinstance(card, dict)
        ) / 10.0

        side_name = "self" if relative_side == 0 else "opp"
        for zone in ("active", "bench", "discard", "prize"):
            for card in player.get(zone) or []:
                _add_card_tokens(vector, side_name, zone, card)
        if relative_side == 0:
            for card in player.get("hand") or []:
                _add_card_tokens(vector, side_name, "hand", card)

    for card in current.get("stadium") or []:
        _add_card_tokens(vector, "global", "stadium", card)
    for card in current.get("looking") or []:
        _add_card_tokens(vector, "self", "looking", card)
    for card in select.get("deck") or []:
        _add_card_tokens(vector, "self", "select_deck", card)

    hbase = STATE_SCALAR_DIM
    _hash_add(vector, hbase, STATE_HASH_DIM, f"select_type:{select.get('type', -1)}")
    _hash_add(vector, hbase, STATE_HASH_DIM, f"select_context:{select.get('context', -1)}")
    _hash_add(
        vector,
        hbase,
        STATE_HASH_DIM,
        f"select_pair:{select.get('type', -1)}:{select.get('context', -1)}",
    )
    for key in ("contextCard", "effect"):
        card_id = _card_id(select.get(key))
        if card_id:
            _hash_add(vector, hbase, STATE_HASH_DIM, f"select_{key}:{card_id}")
    for log in (observation.get("logs") or [])[-20:]:
        if not isinstance(log, dict):
            continue
        player = log.get("playerIndex")
        relative = "self" if player == your_index else "opp" if player == opponent_index else "na"
        log_type = log.get("type", -1)
        _hash_add(vector, hbase, STATE_HASH_DIM, f"log:{relative}:{log_type}")
        for key in ("cardId", "cardIdActive", "cardIdBench", "cardIdBefore", "cardIdAfter", "cardIdTarget"):
            value = log.get(key)
            if isinstance(value, int) and value > 0:
                _hash_add(vector, hbase, STATE_HASH_DIM, f"logcard:{relative}:{log_type}:{value}")
        if isinstance(log.get("attackId"), int):
            _hash_add(vector, hbase, STATE_HASH_DIM, f"logattack:{relative}:{log['attackId']}")
    return vector


def encode_option(
    observation: dict[str, Any],
    option: dict[str, Any],
    option_index: int,
    cards: dict[int, dict[str, Any]],
    attacks: dict[int, dict[str, Any]],
) -> np.ndarray:
    vector = np.zeros(OPTION_DIM, dtype=np.float32)
    select = observation.get("select") or {}
    current = observation.get("current") or {}
    options = select.get("option") or []
    option_type = int(option.get("type", -1))
    your_index = int(current.get("yourIndex", 0))
    owner = option.get("playerIndex")

    vector[0] = 1.0
    vector[1] = option_index / max(1.0, len(options) - 1.0)
    vector[2] = len(options) / 40.0
    vector[3] = _number(select.get("minCount")) / 16.0
    vector[4] = _number(select.get("maxCount")) / 16.0
    if 0 <= option_type <= 16:
        vector[5 + option_type] = 1.0
    vector[22] = _number(option.get("number")) / 20.0
    vector[23] = _number(option.get("area")) / 12.0
    vector[24] = _number(option.get("index")) / 20.0
    vector[25] = 1.0 if owner == your_index else -1.0 if owner == 1 - your_index else 0.0
    vector[26] = _number(option.get("toolIndex")) / 5.0
    vector[27] = _number(option.get("energyIndex")) / 10.0
    vector[28] = _number(option.get("count")) / 10.0
    vector[29] = _number(option.get("inPlayArea")) / 12.0
    vector[30] = _number(option.get("inPlayIndex")) / 5.0
    vector[31] = _number(option.get("attackId")) / 1600.0
    vector[32] = _number(option.get("cardId")) / 1300.0
    vector[33] = _number(option.get("specialConditionType")) / 5.0

    primary, target = resolve_option_cards(observation, option)
    primary_id = _card_id(primary)
    target_id = _card_id(target)
    card = cards.get(primary_id, {})
    vector[34] = primary_id / 1300.0
    vector[35] = _number(card.get("cardType")) / 6.0
    vector[36] = _number(card.get("hp")) / 400.0
    vector[37] = _number(card.get("retreatCost")) / 5.0
    vector[38] = _number(card.get("energyType")) / 11.0
    for offset, key in enumerate(("basic", "stage1", "stage2", "ex", "megaEx", "tera", "aceSpec")):
        vector[39 + offset] = _number(card.get(key))
    vector[46] = len(card.get("attacks") or []) / 3.0
    vector[47] = len(card.get("skills") or []) / 3.0

    attack_id = option.get("attackId") if isinstance(option.get("attackId"), int) else 0
    attack = attacks.get(int(attack_id), {})
    hbase = OPTION_SCALAR_DIM
    context = select.get("context", -1)
    select_type = select.get("type", -1)
    _hash_add(vector, hbase, OPTION_HASH_DIM, f"option_type:{option_type}")
    _hash_add(vector, hbase, OPTION_HASH_DIM, f"context_option:{context}:{option_type}")
    _hash_add(vector, hbase, OPTION_HASH_DIM, f"select_option:{select_type}:{option_type}")
    for key in ("area", "index", "number", "playerIndex", "inPlayArea", "inPlayIndex", "count"):
        if option.get(key) is not None:
            _hash_add(vector, hbase, OPTION_HASH_DIM, f"{option_type}:{key}:{option.get(key)}")
    if primary_id:
        _hash_add(vector, hbase, OPTION_HASH_DIM, f"primary:{primary_id}")
        _hash_add(vector, hbase, OPTION_HASH_DIM, f"context_primary:{context}:{primary_id}")
        _hash_add(vector, hbase, OPTION_HASH_DIM, f"type_primary:{option_type}:{primary_id}")
    if target_id:
        _hash_add(vector, hbase, OPTION_HASH_DIM, f"target:{target_id}")
        _hash_add(vector, hbase, OPTION_HASH_DIM, f"primary_target:{primary_id}:{target_id}")
    if attack_id:
        _hash_add(vector, hbase, OPTION_HASH_DIM, f"attack:{attack_id}")
        _hash_add(vector, hbase, OPTION_HASH_DIM, f"context_attack:{context}:{attack_id}")
        damage_bucket = int(_number(attack.get("damage")) // 30)
        _hash_add(vector, hbase, OPTION_HASH_DIM, f"attack_damage:{damage_bucket}")
        for energy in attack.get("energies") or []:
            _hash_add(vector, hbase, OPTION_HASH_DIM, f"attack_energy:{energy}")

    players = current.get("players") or []
    if 0 <= your_index < len(players):
        active = players[your_index].get("active") or []
        active_id = _card_id(active[0]) if active else 0
        if active_id:
            _hash_add(vector, hbase, OPTION_HASH_DIM, f"active_primary:{active_id}:{primary_id}")
            if attack_id:
                _hash_add(vector, hbase, OPTION_HASH_DIM, f"active_attack:{active_id}:{attack_id}")
    return vector


def build_model_matrix(
    state_features: np.ndarray,
    option_features: np.ndarray,
    option_offsets: np.ndarray,
    decision_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    counts = option_offsets[decision_indices + 1] - option_offsets[decision_indices]
    repeated_state = np.repeat(state_features[decision_indices], counts, axis=0)
    option_indices = np.concatenate(
        [np.arange(option_offsets[i], option_offsets[i + 1]) for i in decision_indices]
    )
    return np.concatenate([repeated_state, option_features[option_indices]], axis=1), option_indices


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def mlp_predict_proba(features: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    hidden = (features - model["scaler_mean"]) / model["scaler_scale"]
    layer_count = int(model["layer_count"][0])
    for layer in range(layer_count):
        hidden = hidden @ model[f"coef_{layer}"] + model[f"intercept_{layer}"]
        if layer < layer_count - 1:
            hidden = np.maximum(hidden, 0.0)
    return sigmoid(hidden.reshape(-1))


def choose_indices(probabilities: np.ndarray, min_count: int, max_count: int, threshold: float) -> list[int]:
    if len(probabilities) == 0 or max_count <= 0:
        return []
    order = np.argsort(-probabilities, kind="stable")
    if min_count == max_count:
        count = min(max_count, len(order))
    else:
        count = int(np.sum(probabilities >= threshold))
        count = max(min_count, min(max_count, count, len(order)))
    return [int(index) for index in order[:count]]

