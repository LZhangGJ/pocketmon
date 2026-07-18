from __future__ import annotations

from typing import Any


STATE_DIM = 32
ACTION_DIM = 18


def _num(value: Any, default: float = 0.0) -> float:
    return float(default if value is None else value)


def _pokemon_summary(player: dict[str, Any]) -> list[float]:
    active = player.get("active") or []
    pokemon = active[0] if active and active[0] else {}
    bench = player.get("bench") or []
    return [
        _num(player.get("deckCount")) / 60.0,
        len(player.get("prize") or []) / 6.0,
        _num(player.get("handCount")) / 20.0,
        len(bench) / max(1.0, _num(player.get("benchMax"), 5)),
        _num(pokemon.get("hp")) / 400.0,
        _num(pokemon.get("maxHp")) / 400.0,
        len(pokemon.get("energies") or []) / 8.0,
        len(pokemon.get("tools") or []) / 4.0,
        sum(_num(card.get("hp")) for card in bench) / 2000.0,
        sum(len(card.get("energies") or []) for card in bench) / 20.0,
        float(bool(player.get("poisoned"))),
        float(any(player.get(name) for name in ("burned", "asleep", "paralyzed", "confused"))),
    ]


def state_features(observation: dict[str, Any]) -> list[float]:
    """Encode only public/current-player information into a stable vector."""
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    players = current.get("players") or [{}, {}]
    your_index = int(current.get("yourIndex", 0) or 0)
    yours = players[your_index] if len(players) > your_index else {}
    theirs = players[1 - your_index] if len(players) > 1 - your_index else {}
    base = [
        _num(current.get("turn")) / 100.0,
        _num(current.get("turnActionCount")) / 30.0,
        float(current.get("firstPlayer") == your_index),
        float(bool(current.get("supporterPlayed"))),
        float(bool(current.get("stadiumPlayed"))),
        float(bool(current.get("energyAttached"))),
        float(bool(current.get("retreated"))),
        _num(select.get("type")) / 32.0,
    ]
    vector = base + _pokemon_summary(yours) + _pokemon_summary(theirs)
    assert len(vector) == STATE_DIM
    return vector


def action_features(option: dict[str, Any], option_index: int, selection_size: int = 1) -> list[float]:
    """Encode an option without assuming that option indices are stable across states."""
    vector = [
        _num(option.get("type")) / 64.0,
        _num(option.get("number")) / 100.0,
        _num(option.get("area")) / 32.0,
        _num(option.get("index")) / 10.0,
        _num(option.get("playerIndex")),
        _num(option.get("toolIndex")) / 4.0,
        _num(option.get("energyIndex")) / 8.0,
        _num(option.get("count")) / 20.0,
        _num(option.get("inPlayArea")) / 32.0,
        _num(option.get("inPlayIndex")) / 10.0,
        _num(option.get("attackId")) / 2000.0,
        _num(option.get("cardId")) / 2000.0,
        _num(option.get("specialConditionType")) / 32.0,
        option_index / 100.0,
        selection_size / 10.0,
        float(option.get("cardId") is not None),
        float(option.get("attackId") is not None),
        1.0,
    ]
    assert len(vector) == ACTION_DIM
    return vector
