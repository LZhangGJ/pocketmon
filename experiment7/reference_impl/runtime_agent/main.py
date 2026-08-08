from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from deck_identity_portable import PortableDeckIdentityTransformerPolicy
from features import encode_option, encode_state, load_catalog
from tokenizer import encode_entities, load_cards


BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path("/kaggle_simulations/agent")
POLICY = PortableDeckIdentityTransformerPolicy(BASE / "deck_identity_bc.npz")
CARDS, ATTACKS = load_catalog(BASE / "engine_catalog.json")
ENTITY_CARDS = load_cards(BASE / "engine_catalog.json")
DECK = np.asarray(
    [int(line) for line in (BASE / "deck.csv").read_text(encoding="utf-8").splitlines() if line.strip()],
    dtype=np.int64,
)
HISTORY_LENGTH = int(POLICY.config["history_length"])
STATE_DIM = int(POLICY.config["state_dim"])
OPTION_DIM = int(POLICY.config["option_dim"])
OPPONENT_CLASSES = int(POLICY.config["opponent_classes"])
HISTORY: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=HISTORY_LENGTH)

_STATS = {
    "modelCalls": 0,
    "forcedCalls": 0,
    "fallbackCalls": 0,
    "truncatedEntityCalls": 0,
    "historyAppends": 0,
    "gameResets": 0,
    "opponentConfidence60Calls": 0,
    **{f"opponentClass{index}Calls": 0 for index in range(OPPONENT_CLASSES)},
}


class _AdvisorStats:
    @staticmethod
    def get_stats() -> dict[str, int]:
        return dict(_STATS)


bc_advisor = _AdvisorStats()


def _bounds(select: dict[str, Any], option_count: int) -> tuple[int, int]:
    minimum = max(0, min(option_count, int(select.get("minCount", 0) or 0)))
    maximum = max(minimum, min(option_count, int(select.get("maxCount", 0) or 0)))
    return minimum, maximum


def _legal(action: list[int], minimum: int, maximum: int, option_count: int) -> bool:
    return (
        minimum <= len(action) <= maximum
        and len(action) == len(set(action))
        and all(
            isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < option_count
            for index in action
        )
    )


def _history_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.zeros((HISTORY_LENGTH, STATE_DIM), dtype=np.float32)
    actions = np.zeros((HISTORY_LENGTH, OPTION_DIM), dtype=np.float32)
    mask = np.zeros(HISTORY_LENGTH, dtype=np.uint8)
    offset = HISTORY_LENGTH - len(HISTORY)
    for slot, (state, action) in enumerate(HISTORY, start=offset):
        states[slot] = state
        actions[slot] = action
        mask[slot] = 1
    return states, actions, mask


def _append_history(
    state: np.ndarray, option_rows: np.ndarray, selected: list[int]
) -> None:
    action = np.zeros(OPTION_DIM, dtype=np.float32)
    if selected:
        action = option_rows[selected].mean(axis=0).astype(np.float32)
    HISTORY.append((state.astype(np.float32, copy=True), action))
    _STATS["historyAppends"] += 1


def _record_opponent_prediction(logits: np.ndarray) -> None:
    shifted = logits - float(np.max(logits))
    probabilities = np.exp(shifted)
    probabilities /= float(np.sum(probabilities))
    prediction = int(np.argmax(probabilities))
    _STATS[f"opponentClass{prediction}Calls"] += 1
    _STATS["opponentConfidence60Calls"] += int(float(probabilities[prediction]) >= 0.60)


def agent(observation: dict[str, Any]) -> list[int]:
    if not observation or observation.get("select") is None:
        HISTORY.clear()
        _STATS["gameResets"] += 1
        return DECK.tolist()

    select = observation.get("select") or {}
    options = select.get("option") or []
    option_count = len(options)
    minimum, maximum = _bounds(select, option_count)
    if option_count == 0:
        return []

    state: np.ndarray | None = None
    option_rows: np.ndarray | None = None
    try:
        state = encode_state(observation)
        option_rows = np.asarray(
            [
                encode_option(observation, option, index, CARDS, ATTACKS)
                for index, option in enumerate(options)
            ],
            dtype=np.float32,
        )
        if minimum == maximum and minimum in (0, option_count):
            action = list(range(minimum))
            _STATS["forcedCalls"] += 1
        else:
            entity_cat, entity_num, entity_mask, truncated = encode_entities(
                observation, ENTITY_CARDS
            )
            history_state, history_action, history_mask = _history_arrays()
            _STATS["modelCalls"] += 1
            _STATS["truncatedEntityCalls"] += int(truncated > 0)
            action, opponent_logits = POLICY.choose(
                state,
                history_state,
                history_action,
                history_mask,
                DECK,
                entity_cat,
                entity_num,
                entity_mask,
                option_rows,
                minimum,
                maximum,
            )
            _record_opponent_prediction(opponent_logits)
        if _legal(action, minimum, maximum, option_count):
            _append_history(state, option_rows, action)
            return action
    except Exception:
        pass

    _STATS["fallbackCalls"] += 1
    action = list(range(minimum))
    if state is not None and option_rows is not None:
        _append_history(state, option_rows, action)
    return action
