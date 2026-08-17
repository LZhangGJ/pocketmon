from __future__ import annotations

import copy
from typing import Any

# The packaged Agent renames the teammate implementation to features_vendor.py.
from features_vendor import OPTION_DIM, STATE_DIM, load_catalog  # type: ignore
import features_vendor as _vendor  # type: ignore

_LAST_OBSERVATION_ID: int | None = None
_LAST_SANITIZED: dict[str, Any] | None = None


def sanitize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Remove hidden card identities while retaining public counts and legal reveals."""

    sanitized = copy.deepcopy(observation)
    current = sanitized.get("current")
    if not isinstance(current, dict):
        return sanitized
    players = current.get("players")
    if not isinstance(players, list):
        return sanitized
    your_index = int(current.get("yourIndex", 0) or 0)
    for player_index, player in enumerate(players):
        if not isinstance(player, dict):
            continue
        prize = player.get("prize")
        if isinstance(prize, list):
            player["prize"] = [{} for _ in prize]
        if player_index != your_index:
            player["hand"] = []
    return sanitized


def _cached_sanitized(observation: dict[str, Any]) -> dict[str, Any]:
    global _LAST_OBSERVATION_ID, _LAST_SANITIZED
    identity = id(observation)
    if identity != _LAST_OBSERVATION_ID or _LAST_SANITIZED is None:
        _LAST_SANITIZED = sanitize_observation(observation)
        _LAST_OBSERVATION_ID = identity
    return _LAST_SANITIZED


def encode_state(observation: dict[str, Any]):
    return _vendor.encode_state(_cached_sanitized(observation))


def encode_option(
    observation: dict[str, Any],
    option: dict[str, Any],
    option_index: int,
    cards,
    attacks,
):
    return _vendor.encode_option(
        _cached_sanitized(observation), option, option_index, cards, attacks
    )
