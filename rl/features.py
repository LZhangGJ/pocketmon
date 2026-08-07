from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


STATE_DIM = 32
ACTION_DIM = 18
HISTORY_DIM = STATE_DIM + ACTION_DIM + 2
ENTITY_DIM = 8
CARD_METADATA_DIM = 17
ATTACK_METADATA_DIM = 14
CARD_TEXT_EMBEDDING_DIM = 128
CARD_VOCAB_SIZE = 2048
ATTACK_VOCAB_SIZE = 2048
ENTITY_ZONE_COUNT = 16
MAX_VISIBLE_ENTITIES = 96
MAX_DECK_SIZE = 60
MAX_CONTEXT_CARDS = 60
RESOURCE_DIM = 8
BELIEF_DIM = 4


# AreaType values from the official SDK. Keeping the integers here avoids a
# runtime dependency on cg.dll in preprocessing and checkpoint inference.
AREA_DECK, AREA_HAND, AREA_DISCARD = 1, 2, 3
AREA_ACTIVE, AREA_BENCH, AREA_PRIZE = 4, 5, 6
AREA_STADIUM, AREA_LOOKING = 7, 12


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


def _safe_id(value: Any, vocabulary_size: int) -> int:
    if isinstance(value, bool):
        return 0
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 0
    return value if 0 < value < vocabulary_size else 0


def _as_cards(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [card for card in value if isinstance(card, dict)]
    return []


def _card_at(observation: dict[str, Any], area: Any, index: Any, player_index: Any) -> dict[str, Any] | None:
    try:
        area, index = int(area), int(index)
    except (TypeError, ValueError):
        return None
    current = observation.get("current") or {}
    select = observation.get("select") or {}
    players = current.get("players") or []
    try:
        player_index = int(player_index)
    except (TypeError, ValueError):
        player_index = int(current.get("yourIndex", 0) or 0)
    player = players[player_index] if 0 <= player_index < len(players) and isinstance(players[player_index], dict) else {}
    collections = {
        AREA_DECK: select.get("deck"),
        AREA_HAND: player.get("hand"),
        AREA_DISCARD: player.get("discard"),
        AREA_ACTIVE: player.get("active"),
        AREA_BENCH: player.get("bench"),
        AREA_PRIZE: player.get("prize"),
        AREA_STADIUM: current.get("stadium"),
        AREA_LOOKING: current.get("looking"),
    }
    cards = collections.get(area)
    if isinstance(cards, dict):
        cards = [cards]
    if not isinstance(cards, list) or not 0 <= index < len(cards):
        return None
    return cards[index] if isinstance(cards[index], dict) else None


def option_identity_features(observation: dict[str, Any], option: dict[str, Any]) -> tuple[int, int, int]:
    """Resolve source-card, target-card and attack IDs from a visible option."""

    current = observation.get("current") or {}
    select = observation.get("select") or {}
    your_index = int(current.get("yourIndex", 0) or 0)
    source = None
    raw_option_type = option.get("type")
    option_type = int(-1 if raw_option_type is None else raw_option_type)
    if option.get("cardId") is not None:
        source_id = _safe_id(option.get("cardId"), CARD_VOCAB_SIZE)
    else:
        source = _card_at(
            observation, option.get("area"), option.get("index"), option.get("playerIndex", your_index)
        )
        # PLAY options specify only the index within the acting player's hand.
        if source is None and option_type == 7:
            source = _card_at(observation, AREA_HAND, option.get("index"), your_index)
        # Attached-card choices identify their parent plus the child index.
        if source is not None and option_type in (4, 5, 6):
            child_key = "tools" if option_type == 4 else "energyCards"
            child_index = option.get("toolIndex") if option_type == 4 else option.get("energyIndex")
            children = source.get(child_key) or []
            if isinstance(child_index, int) and 0 <= child_index < len(children) and isinstance(children[child_index], dict):
                source = children[child_index]
        if source is None and isinstance(select.get("contextCard"), dict):
            source = select["contextCard"]
        source_id = _safe_id(source.get("id") if source else None, CARD_VOCAB_SIZE)

    target = _card_at(observation, option.get("inPlayArea"), option.get("inPlayIndex"), your_index)
    if target is None and option_type == 13:
        target = _card_at(observation, AREA_ACTIVE, 0, your_index)
    target_id = _safe_id(target.get("id") if target else None, CARD_VOCAB_SIZE)
    attack_id = _safe_id(option.get("attackId"), ATTACK_VOCAB_SIZE)
    return source_id, target_id, attack_id


def structured_observation_features(observation: dict[str, Any], options: list[Any]) -> dict[str, Any]:
    """Build visible entities and categorical option identities without hidden cards."""

    current = observation.get("current") or {}
    select = observation.get("select") or {}
    players = current.get("players") or []
    your_index = int(current.get("yourIndex", 0) or 0)
    yours = players[your_index] if 0 <= your_index < len(players) and isinstance(players[your_index], dict) else {}
    other_index = 1 - your_index
    theirs = players[other_index] if 0 <= other_index < len(players) and isinstance(players[other_index], dict) else {}

    entities: list[tuple[int, int, list[float]]] = []

    def add(cards: Any, zone: int) -> None:
        for position, card in enumerate(_as_cards(cards)):
            card_id = _safe_id(card.get("id"), CARD_VOCAB_SIZE)
            if not card_id:
                continue
            hp = _num(card.get("hp"))
            max_hp = _num(card.get("maxHp"))
            numeric = [
                hp / 400.0,
                max_hp / 400.0,
                max(0.0, max_hp - hp) / 400.0,
                len(card.get("energies") or []) / 8.0,
                len(card.get("tools") or []) / 4.0,
                len(card.get("preEvolution") or []) / 4.0,
                float(bool(card.get("appearThisTurn"))),
                position / 60.0,
            ]
            entities.append((card_id, zone, numeric))

    # Opponent hand and prize identities are deliberately excluded. The engine
    # normally represents them as None, but the extractor remains safe even if a
    # malformed replay happens to contain those hidden fields.
    for zone, cards in enumerate((
        yours.get("active"), yours.get("bench"), yours.get("hand"), yours.get("discard"),
        theirs.get("active"), theirs.get("bench"), theirs.get("discard"),
        current.get("stadium"), current.get("looking"), select.get("deck"), select.get("contextCard"),
    )):
        add(cards, zone + 1)
    entities = entities[:MAX_VISIBLE_ENTITIES]

    identities = [
        option_identity_features(observation, option if isinstance(option, dict) else {})
        for option in options
    ]
    return {
        "option_card_ids": [item[0] for item in identities],
        "option_target_card_ids": [item[1] for item in identities],
        "option_attack_ids": [item[2] for item in identities],
        "entity_card_ids": [item[0] for item in entities],
        "entity_zone_ids": [item[1] for item in entities],
        "entity_features": [item[2] for item in entities],
    }


def card_metadata_table(path: Path) -> list[list[float]]:
    table = [[0.0] * CARD_METADATA_DIM for _ in range(CARD_VOCAB_SIZE)]
    for card in json.loads(Path(path).read_text(encoding="utf-8")):
        card_id = _safe_id(card.get("cardId"), CARD_VOCAB_SIZE)
        if not card_id:
            continue
        raw_card_type = card.get("cardType")
        card_type = int(-1 if raw_card_type is None else raw_card_type)
        one_hot = [float(card_type == index) for index in range(7)]
        table[card_id] = one_hot + [
            _num(card.get("hp")) / 400.0,
            _num(card.get("retreatCost")) / 5.0,
            _num(card.get("energyType"), -1) / 12.0,
            *[float(bool(card.get(key))) for key in ("basic", "stage1", "stage2", "ex", "megaEx", "tera", "aceSpec")],
        ]
    return table


def attack_metadata_table(path: Path) -> list[list[float]]:
    table = [[0.0] * ATTACK_METADATA_DIM for _ in range(ATTACK_VOCAB_SIZE)]
    for attack in json.loads(Path(path).read_text(encoding="utf-8")):
        attack_id = _safe_id(attack.get("attackId"), ATTACK_VOCAB_SIZE)
        if not attack_id:
            continue
        energies = attack.get("energies") or []
        histogram = [float(sum(int(value == kind) for value in energies)) / 5.0 for kind in range(12)]
        table[attack_id] = [_num(attack.get("damage")) / 400.0, len(energies) / 5.0, *histogram]
    return table


def _hashed_text_vector(text: str, dimension: int = CARD_TEXT_EMBEDDING_DIM) -> list[float]:
    """Return a deterministic signed-hash embedding for card-effect text.

    Word/bigram features capture shared rules language, while character n-grams
    remain useful when the official database contains legacy mojibake.  The
    fixed transform keeps checkpoints self-contained and reproducible without
    downloading a private embedding model at inference time.
    """

    if dimension <= 0:
        raise ValueError("text embedding dimension must be positive")
    normalized = " ".join(re.findall(r"[a-z0-9]+|\{[^}]+\}", str(text).casefold()))
    words = normalized.split()
    features = [f"w:{word}" for word in words]
    features.extend(f"b:{left}_{right}" for left, right in zip(words, words[1:]))
    compact = normalized.replace(" ", "_")
    features.extend(f"c3:{compact[index:index + 3]}" for index in range(max(0, len(compact) - 2)))
    vector = [0.0] * dimension
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8, person=b"ptcg-text-v1").digest()
        bucket = int.from_bytes(digest[:4], "little") % dimension
        vector[bucket] += 1.0 if digest[4] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    return vector if norm == 0.0 else [value / norm for value in vector]


def card_text_embedding_table(
    card_path: Path,
    attack_path: Path,
    dimension: int = CARD_TEXT_EMBEDDING_DIM,
) -> list[list[float]]:
    """Embed official card names, effect text and referenced attack text."""

    attacks = {
        _safe_id(attack.get("attackId"), ATTACK_VOCAB_SIZE): attack
        for attack in json.loads(Path(attack_path).read_text(encoding="utf-8"))
    }
    table = [[0.0] * dimension for _ in range(CARD_VOCAB_SIZE)]
    for card in json.loads(Path(card_path).read_text(encoding="utf-8")):
        card_id = _safe_id(card.get("cardId"), CARD_VOCAB_SIZE)
        if not card_id:
            continue
        fragments = [str(card.get("name") or "")]
        for skill in card.get("skills") or []:
            if isinstance(skill, dict):
                fragments.extend((str(skill.get("name") or ""), str(skill.get("text") or "")))
        for attack_id in card.get("attacks") or []:
            attack = attacks.get(_safe_id(attack_id, ATTACK_VOCAB_SIZE)) or {}
            fragments.extend((str(attack.get("name") or ""), str(attack.get("text") or "")))
        table[card_id] = _hashed_text_vector(" ".join(fragments), dimension)
    return table


def _visible_card_counter(player: dict[str, Any], *, include_private: bool) -> Counter[int]:
    """Count card identities visible to the acting player, including attachments."""

    zones = ["active", "bench", "discard"]
    if include_private:
        zones.extend(("hand", "prize"))
    visible: Counter[int] = Counter()
    for zone in zones:
        for card in _as_cards(player.get(zone)):
            card_id = _safe_id(card.get("id"), CARD_VOCAB_SIZE)
            if card_id:
                visible[card_id] += 1
            for key in ("energyCards", "tools", "preEvolution"):
                for attached in _as_cards(card.get(key)):
                    attached_id = _safe_id(attached.get("id"), CARD_VOCAB_SIZE)
                    if attached_id:
                        visible[attached_id] += 1
    return visible


def _remaining_multiset(deck: list[int], known: Counter[int]) -> list[int]:
    counts = Counter(_safe_id(card_id, CARD_VOCAB_SIZE) for card_id in deck)
    counts.pop(0, None)
    remaining: list[int] = []
    for card_id in sorted(counts):
        remaining.extend([card_id] * max(0, counts[card_id] - known[card_id]))
    return remaining[:MAX_CONTEXT_CARDS]


def _normalized_entropy(probabilities: list[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(value * math.log(max(value, 1e-12)) for value in probabilities)
    return entropy / math.log(len(probabilities))


def opponent_deck_belief(
    observation: dict[str, Any],
    prototypes: list[dict[str, Any]] | None,
) -> tuple[list[int], list[float], dict[str, Any]]:
    """Infer a frozen deck prototype from public opponent cards only.

    The returned deck is a belief context, not the hidden submitted deck.  A
    combinatorial likelihood rewards prototypes capable of producing the
    observed multiset, while impossible copy counts receive a finite penalty.
    """

    prototypes = list(prototypes or [])
    if not prototypes:
        return [], [0.0] * BELIEF_DIM, {"prototype": None, "confidence": 0.0}
    current = observation.get("current") or {}
    players = current.get("players") or []
    your_index = int(current.get("yourIndex", 0) or 0)
    opponent = players[1 - your_index] if len(players) > 1 and isinstance(players[1 - your_index], dict) else {}
    observed = _visible_card_counter(opponent, include_private=False)
    observed_total = sum(observed.values())
    logits: list[float] = []
    normalized: list[tuple[str, list[int], float]] = []
    for index, item in enumerate(prototypes):
        deck = [
            _safe_id(card_id, CARD_VOCAB_SIZE)
            for card_id in item.get("deck", [])
        ]
        if len(deck) != MAX_DECK_SIZE or not all(deck):
            raise ValueError(f"opponent prototype {index} must contain 60 valid card ids")
        counts = Counter(deck)
        log_likelihood = math.log(max(float(item.get("prior", 1.0)), 1e-12))
        for card_id, seen in observed.items():
            available = counts[card_id]
            if seen > available:
                log_likelihood -= 30.0 * (seen - available)
            elif seen:
                log_likelihood += math.lgamma(available + 1) - math.lgamma(seen + 1) - math.lgamma(available - seen + 1)
        normalized.append((str(item.get("name") or f"prototype_{index:03d}"), deck, log_likelihood))
        logits.append(log_likelihood)
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    denominator = sum(weights)
    probabilities = [value / denominator for value in weights]
    best_index = max(range(len(probabilities)), key=lambda index: (probabilities[index], -index))
    name, best_deck, _ = normalized[best_index]
    best_counts = Counter(best_deck)
    covered = sum(min(count, best_counts[card_id]) for card_id, count in observed.items())
    confidence = probabilities[best_index]
    features = [
        confidence,
        _normalized_entropy(probabilities),
        observed_total / 60.0,
        covered / max(1, observed_total),
    ]
    return sorted(best_deck), features, {
        "prototype": name,
        "confidence": confidence,
        "entropy": features[1],
        "observed_cards": observed_total,
        "coverage": features[3],
    }


def enhanced_context_features(
    observation: dict[str, Any],
    options: list[Any],
    deck: list[int],
    opponent_prototypes: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Build leakage-safe remaining-card, draw-probability and belief inputs."""

    if len(deck) != MAX_DECK_SIZE:
        raise ValueError("enhanced context requires a 60-card acting deck")
    current = observation.get("current") or {}
    players = current.get("players") or []
    your_index = int(current.get("yourIndex", 0) or 0)
    yours = players[your_index] if len(players) > your_index and isinstance(players[your_index], dict) else {}
    known = _visible_card_counter(yours, include_private=True)
    remaining = _remaining_multiset(deck, known)
    remaining_counts = Counter(remaining)
    option_ids = {
        option_identity_features(observation, option if isinstance(option, dict) else {})[0]
        for option in options
    }
    option_ids.discard(0)
    unknown_prizes = sum(card is None for card in (yours.get("prize") or []))
    deck_count = max(0, int(_num(yours.get("deckCount"))))
    top_count = max(remaining_counts.values(), default=0)
    option_count = sum(remaining_counts[card_id] for card_id in option_ids)
    resource = [
        deck_count / 60.0,
        len(remaining) / 60.0,
        unknown_prizes / 6.0,
        sum(known.values()) / 60.0,
        len(remaining_counts) / 60.0,
        deck_count / max(1, len(remaining)),
        top_count / max(1, len(remaining)),
        option_count / max(1, len(remaining)),
    ]
    belief_deck, belief, audit = opponent_deck_belief(observation, opponent_prototypes)
    assert len(resource) == RESOURCE_DIM and len(belief) == BELIEF_DIM
    return {
        "remaining_card_ids": remaining,
        "resource_features": resource,
        "opponent_belief_card_ids": belief_deck,
        "opponent_belief_features": belief,
        "opponent_belief_audit": audit,
    }
def history_features(state: list[float], options: list[list[float]], action: list[int]) -> list[float]:
    """Encode one completed prior decision without using any later-row information."""

    selected = [options[index] for index in action]
    if selected:
        option_summary = [sum(values) / len(selected) for values in zip(*selected)]
    else:
        option_summary = [0.0] * ACTION_DIM
    vector = list(state) + option_summary + [len(action) / 10.0, float(not action)]
    assert len(vector) == HISTORY_DIM
    return vector
