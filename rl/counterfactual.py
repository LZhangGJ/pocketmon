from __future__ import annotations

import dataclasses
import random
from collections import Counter
from typing import Any

import torch

from .features import history_features
from .ppo import model_row_for_model, predict_state_value, rank_single_actions, sample_action


def _card_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        raw = value.get("id", value.get("cardId"))
    else:
        raw = getattr(value, "id", getattr(value, "cardId", None))
    return int(raw) if isinstance(raw, int) and not isinstance(raw, bool) else None


def _zone_cards(player: dict[str, Any]) -> Counter[int]:
    visible: Counter[int] = Counter()
    for zone in ("hand", "discard", "active", "bench", "prize"):
        for card in player.get(zone) or []:
            card_id = _card_id(card)
            if card_id is not None:
                visible[card_id] += 1
            if isinstance(card, dict):
                for attachment in (card.get("energyCards") or []) + (card.get("tools") or []):
                    attachment_id = _card_id(attachment)
                    if attachment_id is not None:
                        visible[attachment_id] += 1
    return visible


def _remaining(deck: list[int], visible: Counter[int]) -> list[int]:
    counts = Counter(deck)
    result = []
    for card_id, count in counts.items():
        result.extend([card_id] * max(0, count - visible[card_id]))
    return result


def sample_hidden_zones(
    observation: dict[str, Any],
    decks: list[list[int]],
    basic_ids: set[int],
    rng: random.Random,
) -> dict[str, list[int]]:
    """Determinize hidden zones using the exact training matchup decks."""

    current = observation.get("current") or {}
    players = current.get("players") or []
    if len(players) != 2:
        raise ValueError("counterfactual observation needs two players")
    me_index = int(current["yourIndex"])
    me, opponent = players[me_index], players[1 - me_index]

    def partition(player: dict[str, Any], deck: list[int], include_hand: bool) -> tuple[list[int], list[int], list[int]]:
        pool = _remaining(deck, _zone_cards(player))
        hidden_prizes = sum(card is None for card in (player.get("prize") or []))
        need = int(player.get("deckCount", 0)) + hidden_prizes + (int(player.get("handCount", 0)) if include_hand else 0)
        padding = next((card_id for card_id in deck if card_id in basic_ids), deck[0])
        pool.extend([padding] * max(0, need - len(pool)))
        rng.shuffle(pool)
        deck_count = int(player.get("deckCount", 0))
        hidden_deck = pool[:deck_count]
        offset = deck_count
        prize_fill = iter(pool[offset:offset + hidden_prizes])
        prizes = []
        for card in (player.get("prize") or []):
            value = _card_id(card) if card is not None else next(prize_fill, padding)
            prizes.append(padding if value is None else value)
        offset += hidden_prizes
        hand = pool[offset:offset + int(player.get("handCount", 0))] if include_hand else []
        return hidden_deck, [int(value) for value in prizes], hand

    your_deck, your_prize, _ = partition(me, decks[me_index], False)
    opponent_deck, opponent_prize, opponent_hand = partition(opponent, decks[1 - me_index], True)
    active = opponent.get("active") or []
    opponent_active = []
    if active and active[0] is None:
        opponent_active = [next((card_id for card_id in decks[1 - me_index] if card_id in basic_ids), decks[1 - me_index][0])]
    return {
        "your_deck": your_deck,
        "your_prize": your_prize,
        "opponent_deck": opponent_deck,
        "opponent_prize": opponent_prize,
        "opponent_hand": opponent_hand,
        "opponent_active": opponent_active,
    }


def _as_dict(observation: Any) -> dict[str, Any]:
    return dataclasses.asdict(observation) if dataclasses.is_dataclass(observation) else dict(observation)


def terminal_value(observation: dict[str, Any], root_player: int) -> float | None:
    result = int((observation.get("current") or {}).get("result", -1))
    if result == -1:
        return None
    return 0.0 if result == 2 else (1.0 if result == root_player else -1.0)


@torch.inference_mode()
def counterfactual_action_values(
    *,
    model,
    observation: dict[str, Any],
    decks: list[list[int]],
    basic_ids: set[int],
    device: torch.device,
    rng: random.Random,
    candidates: int = 4,
    determinizations: int = 2,
    horizon: int = 16,
    history: list[list[float]] | None = None,
) -> list[dict[str, float | int]]:
    """Estimate Q(s,a) by branching official Search API states."""

    from cg.api import search_begin, search_end, search_step, to_observation_class

    root_player = int((observation.get("current") or {})["yourIndex"])
    ranked = rank_single_actions(
        model, observation, decks[root_player], device, candidates, history=history
    )
    if len(ranked) < 2:
        return []
    totals = {index: [] for index in ranked}
    obs_class = to_observation_class(observation)
    for _ in range(determinizations):
        hidden = sample_hidden_zones(observation, decks, basic_ids, rng)
        began = False
        try:
            root = search_begin(obs_class, **hidden)
            began = True
            for option_index in ranked:
                branch = search_step(root.searchId, [option_index])
                current = _as_dict(branch.observation)
                search_id = branch.searchId
                branch_histories = {root_player: list(history or []), 1 - root_player: []}
                root_row = model_row_for_model(
                    model, observation, decks[root_player], [option_index], branch_histories[root_player]
                )
                branch_histories[root_player].append(history_features(
                    root_row["state"], root_row["options"], [option_index]
                ))
                for _step in range(horizon):
                    terminal = terminal_value(current, root_player)
                    if terminal is not None:
                        break
                    player = int((current.get("current") or {})["yourIndex"])
                    action, _, _, _ = sample_action(
                        model,
                        current,
                        decks[player],
                        device,
                        temperature=0.7,
                        history=branch_histories[player],
                    )
                    action_row = model_row_for_model(
                        model, current, decks[player], action, branch_histories[player]
                    )
                    branch_histories[player].append(history_features(
                        action_row["state"], action_row["options"], action
                    ))
                    maximum = int(getattr(model, "history_length", 0) or 0)
                    if maximum:
                        branch_histories[player] = branch_histories[player][-maximum:]
                    branch = search_step(search_id, action)
                    current, search_id = _as_dict(branch.observation), branch.searchId
                target = terminal_value(current, root_player)
                if target is None:
                    player = int((current.get("current") or {})["yourIndex"])
                    bootstrap = predict_state_value(
                        model, current, decks[player], device, history=branch_histories[player]
                    )
                    target = bootstrap if player == root_player else -bootstrap
                totals[option_index].append(float(max(-1.0, min(1.0, target))))
        finally:
            if began:
                search_end()
    result = []
    for option_index in ranked:
        values = totals[option_index]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        result.append({"option_index": option_index, "target": mean, "target_std": variance ** 0.5, "samples": len(values)})
    return result
