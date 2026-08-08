from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np


MAX_ENTITIES = 96
ENTITY_CAT_DIM = 10
ENTITY_NUM_DIM = 12

# zone, side, kind and host location are deliberately relative to the actor.
ZONE = {
    "self_active": 1,
    "self_bench": 2,
    "self_hand": 3,
    "self_discard": 4,
    "self_prize": 5,
    "opp_active": 6,
    "opp_bench": 7,
    "opp_discard": 8,
    "opp_prize": 9,
    "stadium": 10,
    "looking": 11,
    "select_deck": 12,
    "energy": 13,
    "tool": 14,
    "pre_evolution": 15,
}

KIND_CARD = 1
KIND_POKEMON = 2
KIND_ENERGY = 3
KIND_TOOL = 4
KIND_PRE_EVOLUTION = 5


def load_cards(path: str | Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(card["cardId"]): card for card in payload["cards"]}


def _card_id(card: Any) -> int:
    if isinstance(card, dict) and isinstance(card.get("id"), int):
        return int(card["id"])
    return 0


def _stage(card_meta: dict[str, Any]) -> int:
    if card_meta.get("stage2"):
        return 3
    if card_meta.get("stage1"):
        return 2
    if card_meta.get("basic"):
        return 1
    return 0


def _is_pokemon(card: Any) -> bool:
    return isinstance(card, dict) and isinstance(card.get("hp"), (int, float))


def encode_entities(
    observation: dict[str, Any],
    cards: dict[int, dict[str, Any]],
    max_entities: int = MAX_ENTITIES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Encode only actor-visible card entities; hidden opponent cards never enter."""

    cat = np.zeros((max_entities, ENTITY_CAT_DIM), dtype=np.int16)
    num = np.zeros((max_entities, ENTITY_NUM_DIM), dtype=np.float32)
    mask = np.zeros(max_entities, dtype=np.uint8)
    entries: list[tuple[Any, ...]] = []

    def add(
        card: Any,
        zone: int,
        side: int,
        slot: int,
        kind: int | None = None,
        host_zone: int = 0,
        host_slot: int = 0,
    ) -> None:
        card_id = _card_id(card)
        if card_id <= 0 or not isinstance(card, dict):
            return
        meta = cards.get(card_id, {})
        pokemon = _is_pokemon(card)
        resolved_kind = kind if kind is not None else (KIND_POKEMON if pokemon else KIND_CARD)
        card_type = int(meta.get("cardType", -1)) + 1
        energy_type = int(meta.get("energyType", -1)) + 1
        entries.append(
            (
                card_id,
                zone,
                side,
                resolved_kind,
                min(max(int(slot), 0), 63) + 1,
                host_zone,
                min(max(int(host_slot), 0), 63) + 1 if host_zone else 0,
                max(card_type, 0),
                max(energy_type, 0),
                _stage(meta),
                float(card.get("hp", 0.0)) / 400.0,
                float(card.get("maxHp", 0.0)) / 400.0,
                max(0.0, float(card.get("maxHp", 0.0)) - float(card.get("hp", 0.0))) / 400.0,
                float(bool(card.get("appearThisTurn"))),
                len(card.get("energies") or []) / 10.0,
                len(card.get("tools") or []) / 4.0,
                len(card.get("preEvolution") or []) / 3.0,
                float(bool(meta.get("ex"))),
                float(bool(meta.get("tera"))),
                float(bool(meta.get("aceSpec"))),
                float(meta.get("retreatCost", 0.0)) / 5.0,
                float(meta.get("hp", 0.0)) / 400.0,
            )
        )

        # Attachment tokens let attention distinguish exact resources on a host.
        if pokemon:
            for child_slot, child in enumerate(card.get("energyCards") or []):
                add(child, ZONE["energy"], side, child_slot, KIND_ENERGY, zone, slot)
            for child_slot, child in enumerate(card.get("tools") or []):
                add(child, ZONE["tool"], side, child_slot, KIND_TOOL, zone, slot)
            for child_slot, child in enumerate(card.get("preEvolution") or []):
                add(child, ZONE["pre_evolution"], side, child_slot, KIND_PRE_EVOLUTION, zone, slot)

    current = observation.get("current") or {}
    players = current.get("players") or []
    your_index = int(current.get("yourIndex", 0))
    opponent_index = 1 - your_index

    # Important live entities come first so the cap can only truncate old discard cards.
    for relative_side, player_index, prefix in (
        (0, your_index, "self"),
        (1, opponent_index, "opp"),
    ):
        if not (0 <= player_index < len(players)):
            continue
        player = players[player_index]
        for slot, card in enumerate(player.get("active") or []):
            add(card, ZONE[f"{prefix}_active"], relative_side, slot)
        for slot, card in enumerate(player.get("bench") or []):
            add(card, ZONE[f"{prefix}_bench"], relative_side, slot)
        if relative_side == 0:  # opponent hand is hidden and must stay hidden
            for slot, card in enumerate(player.get("hand") or []):
                add(card, ZONE["self_hand"], relative_side, slot)

    for slot, card in enumerate(current.get("stadium") or []):
        add(card, ZONE["stadium"], 2, slot)
    for slot, card in enumerate(current.get("looking") or []):
        add(card, ZONE["looking"], 0, slot)
    select = observation.get("select") or {}
    for slot, card in enumerate(select.get("deck") or []):
        add(card, ZONE["select_deck"], 0, slot)

    for relative_side, player_index, prefix in (
        (0, your_index, "self"),
        (1, opponent_index, "opp"),
    ):
        if not (0 <= player_index < len(players)):
            continue
        discard = players[player_index].get("discard") or []
        # Keep the most recent discard cards when the entity budget is tight.
        for slot, card in enumerate(discard[-24:]):
            add(card, ZONE[f"{prefix}_discard"], relative_side, slot)

    truncated = max(0, len(entries) - max_entities)
    for index, entry in enumerate(entries[:max_entities]):
        cat[index] = np.asarray(entry[:ENTITY_CAT_DIM], dtype=np.int16)
        num[index] = np.asarray(entry[ENTITY_CAT_DIM:], dtype=np.float32)
        mask[index] = 1
    return cat, num, mask, truncated


def _zone_cards(current: dict[str, Any], player_index: int, zone: str) -> list[Any]:
    players = current.get("players") or []
    if not (0 <= player_index < len(players)):
        return []
    value = players[player_index].get(zone)
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
    if area == 1:
        values = select.get("deck") or []
    elif area == 2:
        values = _zone_cards(current, owner, "hand")
    elif area == 3:
        values = _zone_cards(current, owner, "discard")
    elif area == 4:
        values = _zone_cards(current, owner, "active")
    elif area == 5:
        values = _zone_cards(current, owner, "bench")
    elif area == 6:
        values = _zone_cards(current, owner, "prize")
    elif area == 7:
        values = current.get("stadium") or []
    elif area == 12:
        values = current.get("looking") or []
    else:
        values = []
    if index >= len(values) or not isinstance(values[index], dict):
        return None
    return values[index]


def resolve_option_cards(
    observation: dict[str, Any], option: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    option_type = option.get("type")
    current = observation.get("current") or {}
    your_index = int(current.get("yourIndex", 0))
    primary: dict[str, Any] | None = None
    target: dict[str, Any] | None = None
    if option_type == 7:
        primary = _resolve_area_card(observation, 2, option.get("index"), your_index)
    elif option_type in (8, 9):
        primary = _resolve_area_card(observation, option.get("area"), option.get("index"), your_index)
        target = _resolve_area_card(
            observation, option.get("inPlayArea"), option.get("inPlayIndex"), your_index
        )
    elif option_type in (10, 11):
        primary = _resolve_area_card(observation, option.get("area"), option.get("index"), your_index)
    elif option_type in (3, 4, 5, 6):
        target = _resolve_area_card(
            observation, option.get("area"), option.get("index"), option.get("playerIndex")
        )
        primary = target
        if option_type == 4 and target is not None:
            values = target.get("tools") or []
            idx = option.get("toolIndex")
            if isinstance(idx, int) and 0 <= idx < len(values) and isinstance(values[idx], dict):
                primary = values[idx]
        elif option_type in (5, 6) and target is not None:
            values = target.get("energyCards") or []
            idx = option.get("energyIndex")
            if isinstance(idx, int) and 0 <= idx < len(values) and isinstance(values[idx], dict):
                primary = values[idx]
    elif option_type == 15 and isinstance(option.get("cardId"), int):
        primary = {"id": option["cardId"]}
    return primary, target


def semantic_option_signature(observation: dict[str, Any], option: dict[str, Any]) -> tuple[Any, ...]:
    """Ignore copy indices but preserve distinct in-play targets."""

    current = observation.get("current") or {}
    your_index = int(current.get("yourIndex", 0))
    owner = option.get("playerIndex")
    relative_owner = 0 if owner == your_index else 1 if owner == 1 - your_index else 2
    primary, target = resolve_option_cards(observation, option)
    target_serial = int(target.get("serial", 0)) if _is_pokemon(target) else 0
    return (
        int(option.get("type", -1)),
        int(option.get("area", -1)),
        int(option.get("inPlayArea", -1)),
        int(option.get("inPlayIndex", -1)),
        int(option.get("number", -1)),
        int(option.get("count", -1)),
        int(option.get("attackId", -1)),
        int(option.get("cardId", -1)),
        int(option.get("specialConditionType", -1)),
        relative_owner,
        _card_id(primary),
        _card_id(target),
        target_serial,
    )


def semantic_option_hash(observation: dict[str, Any], option: dict[str, Any]) -> np.uint32:
    payload = json.dumps(semantic_option_signature(observation, option), separators=(",", ":"))
    return np.uint32(zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF)


def expand_semantic_labels(labels: np.ndarray, semantic_hashes: np.ndarray) -> np.ndarray:
    expanded = np.asarray(labels, dtype=np.float32).copy()
    positive_hashes = set(int(value) for value in semantic_hashes[np.asarray(labels, dtype=bool)])
    if positive_hashes:
        expanded = np.asarray([float(int(value) in positive_hashes) for value in semantic_hashes], dtype=np.float32)
    return expanded
