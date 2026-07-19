from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal


Alignment = Literal["previous", "same"]


@dataclass(frozen=True)
class ReplayTransition:
    episode_id: str
    player: int
    action_step: int
    observation_step: int
    action: Any
    observation: dict[str, Any] | None
    status: Any


@dataclass(frozen=True)
class ActionValidation:
    valid: bool
    kind: Literal["decision", "setup", "invalid"]
    reason: str
    option_count: int | None = None
    min_count: int | None = None
    max_count: int | None = None


def replay_episode_id(replay: dict[str, Any], fallback: str = "unknown") -> str:
    info = replay.get("info") or {}
    for key in ("EpisodeId", "episode_id", "id"):
        value = info.get(key)
        if value is not None:
            return str(value)
    return fallback


def replay_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _player_view(steps: list[Any], step_index: int, player: int) -> dict[str, Any] | None:
    if step_index < 0 or step_index >= len(steps):
        return None
    step = steps[step_index]
    if not isinstance(step, list) or player >= len(step) or not isinstance(step[player], dict):
        return None
    return step[player]


def iter_transitions(replay: dict[str, Any], alignment: Alignment) -> Iterator[ReplayTransition]:
    """Yield every recorded non-null action with one candidate observation alignment.

    Kaggle simulation replays commonly store the action that produced a step next to
    the post-action observation. DATA-001 compares that ``previous`` interpretation
    against the naïve ``same``-step interpretation instead of assuming either one.
    """

    steps = replay.get("steps") or []
    episode_id = replay_episode_id(replay)
    for action_step, step in enumerate(steps):
        if not isinstance(step, list):
            continue
        for player, action_view in enumerate(step):
            if not isinstance(action_view, dict) or action_view.get("action") is None:
                continue
            observation_step = action_step - 1 if alignment == "previous" else action_step
            observation_view = _player_view(steps, observation_step, player)
            observation = None if observation_view is None else observation_view.get("observation")
            if not isinstance(observation, dict):
                observation = None
            yield ReplayTransition(
                episode_id=episode_id,
                player=player,
                action_step=action_step,
                observation_step=observation_step,
                action=action_view.get("action"),
                observation=observation,
                status=action_view.get("status"),
            )


def validate_transition(transition: ReplayTransition) -> ActionValidation:
    action = transition.action
    if not isinstance(action, list) or any(not isinstance(index, int) for index in action):
        return ActionValidation(False, "invalid", "action_not_integer_list")

    observation = transition.observation
    if observation is None:
        return ActionValidation(False, "invalid", "missing_observation")

    select = observation.get("select")
    if select is None:
        if len(action) == 60:
            return ActionValidation(True, "setup", "deck_submission")
        return ActionValidation(False, "invalid", "missing_select")
    if not isinstance(select, dict):
        return ActionValidation(False, "invalid", "select_not_object")

    options = select.get("option")
    if not isinstance(options, list):
        return ActionValidation(False, "invalid", "options_not_list")
    option_count = len(options)
    try:
        min_count = int(select.get("minCount", 0))
        max_count = int(select.get("maxCount", min_count))
    except (TypeError, ValueError):
        return ActionValidation(False, "invalid", "invalid_selection_bounds", option_count=option_count)

    if min_count < 0 or max_count < min_count:
        return ActionValidation(
            False,
            "invalid",
            "invalid_selection_bounds",
            option_count,
            min_count,
            max_count,
        )
    if not min_count <= len(action) <= max_count:
        return ActionValidation(
            False,
            "invalid",
            "selection_count_out_of_bounds",
            option_count,
            min_count,
            max_count,
        )
    if len(set(action)) != len(action):
        return ActionValidation(
            False,
            "invalid",
            "duplicate_option_index",
            option_count,
            min_count,
            max_count,
        )
    if any(index < 0 or index >= option_count for index in action):
        return ActionValidation(
            False,
            "invalid",
            "option_index_out_of_range",
            option_count,
            min_count,
            max_count,
        )
    return ActionValidation(True, "decision", "valid", option_count, min_count, max_count)


def terminal_winner(replay: dict[str, Any]) -> int | None:
    for step in reversed(replay.get("steps") or []):
        if not isinstance(step, list):
            continue
        for view in step:
            if not isinstance(view, dict):
                continue
            observation = view.get("observation") or {}
            current = observation.get("current") or {}
            result = current.get("result")
            if isinstance(result, int) and result != -1:
                return result
    return None


def player_outcome(winner: int | None, player: int) -> float | None:
    if winner is None:
        return None
    if winner == 2:
        return 0.0
    return 1.0 if winner == player else -1.0


def model_observation(observation: dict[str, Any], keep_logs: bool = False) -> dict[str, Any]:
    """Copy one player's view and remove event logs from model input by default."""

    output = copy.deepcopy(observation)
    if not keep_logs:
        output.pop("logs", None)
    return output


def audit_replay(
    replay: dict[str, Any],
    alignment: Alignment,
    *,
    max_examples: int = 20,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for transition in iter_transitions(replay, alignment):
        counts["actions"] += 1
        validation = validate_transition(transition)
        counts[validation.kind] += 1
        reasons[validation.reason] += 1
        if not validation.valid and len(examples) < max_examples:
            examples.append(
                {
                    "episode_id": transition.episode_id,
                    "player": transition.player,
                    "action_step": transition.action_step,
                    "observation_step": transition.observation_step,
                    "action": transition.action,
                    **asdict(validation),
                }
            )
    decisions = counts["decision"] + counts["invalid"]
    return {
        "alignment": alignment,
        "actions": counts["actions"],
        "valid_decisions": counts["decision"],
        "setup_actions": counts["setup"],
        "invalid_decisions": counts["invalid"],
        "valid_rate": counts["decision"] / decisions if decisions else 0.0,
        "reasons": dict(sorted(reasons.items())),
        "examples": examples,
    }


def merge_audits(reports: list[dict[str, Any]], alignment: Alignment, max_examples: int = 20) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for report in reports:
        for key in ("actions", "valid_decisions", "setup_actions", "invalid_decisions"):
            totals[key] += int(report.get(key, 0))
        reasons.update(report.get("reasons") or {})
        examples.extend((report.get("examples") or [])[: max_examples - len(examples)])
    decisions = totals["valid_decisions"] + totals["invalid_decisions"]
    return {
        "alignment": alignment,
        **{key: totals[key] for key in ("actions", "valid_decisions", "setup_actions", "invalid_decisions")},
        "valid_rate": totals["valid_decisions"] / decisions if decisions else 0.0,
        "reasons": dict(sorted(reasons.items())),
        "examples": examples,
    }


def canonical_rows(
    replay: dict[str, Any],
    *,
    alignment: Alignment,
    source_path: str,
    source_sha256: str,
    manifest: dict[str, Any] | None = None,
    policy_source: Literal["winners", "all"] = "winners",
    keep_logs: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    winner = terminal_winner(replay)
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    setup_actions = 0
    episode_id = replay_episode_id(replay, Path(source_path).stem)
    for transition in iter_transitions(replay, alignment):
        validation = validate_transition(transition)
        if validation.kind == "setup":
            setup_actions += 1
            continue
        if not validation.valid:
            invalid.append(
                {
                    "player": transition.player,
                    "action_step": transition.action_step,
                    "observation_step": transition.observation_step,
                    "action": transition.action,
                    **asdict(validation),
                }
            )
            continue

        assert transition.observation is not None
        select = transition.observation["select"]
        options = select["option"]
        outcome = player_outcome(winner, transition.player)
        policy_weight = 1.0 if policy_source == "all" or outcome == 1.0 else 0.0
        rows.append(
            {
                "schema_version": 1,
                "episode_id": episode_id,
                "source_path": source_path,
                "source_sha256": source_sha256,
                "player": transition.player,
                "action_step": transition.action_step,
                "observation_step": transition.observation_step,
                "status": transition.status,
                "winner": winner,
                "outcome": outcome,
                "policy_weight": policy_weight,
                "value_weight": 1.0 if outcome is not None else 0.0,
                "action": transition.action,
                "chosen_options": [options[index] for index in transition.action],
                "observation": model_observation(transition.observation, keep_logs=keep_logs),
                "manifest": manifest or {},
            }
        )
    return rows, {
        "episode_id": episode_id,
        "winner": winner,
        "rows": len(rows),
        "setup_actions": setup_actions,
        "invalid_decisions": len(invalid),
        "invalid_examples": invalid[:20],
    }


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
