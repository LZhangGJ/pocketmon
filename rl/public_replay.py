from __future__ import annotations

import copy
import hashlib
import json
import math
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
    action_status: Any
    observation_status: Any
    submission_status: Any

    @property
    def status(self) -> Any:
        """Backward-compatible alias for the status when the action was requested."""

        return self.submission_status


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


def _transition_at(
    steps: list[Any],
    *,
    episode_id: str,
    player: int,
    action_step: int,
    observation_step: int,
) -> ReplayTransition:
    action_view = _player_view(steps, action_step, player)
    observation_view = _player_view(steps, observation_step, player)
    submission_view = _player_view(steps, action_step - 1, player)
    observation = None if observation_view is None else observation_view.get("observation")
    if not isinstance(observation, dict):
        observation = None
    return ReplayTransition(
        episode_id=episode_id,
        player=player,
        action_step=action_step,
        observation_step=observation_step,
        action=None if action_view is None else action_view.get("action"),
        observation=observation,
        action_status=None if action_view is None else action_view.get("status"),
        observation_status=None if observation_view is None else observation_view.get("status"),
        submission_status=None if submission_view is None else submission_view.get("status"),
    )


def iter_transitions(
    replay: dict[str, Any],
    alignment: Alignment,
    *,
    acting_only: bool = True,
) -> Iterator[ReplayTransition]:
    """Yield recorded actions with one candidate observation alignment.

    Kaggle Environments requests actions from ``steps[t - 1]`` and appends the
    interpreter's resulting state, including those actions, as ``steps[t]``.  The
    action eligibility status is therefore also the status at ``t - 1``; the status
    stored beside the action is post-interpreter state and must not be used to decide
    whether the player acted.  ``acting_only=False`` is intended for diagnostics that
    need to count reset and inactive placeholders explicitly.
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
            transition = _transition_at(
                steps,
                episode_id=episode_id,
                player=player,
                action_step=action_step,
                observation_step=observation_step,
            )
            if acting_only and transition.submission_status != "ACTIVE":
                continue
            yield transition


def validate_transition(transition: ReplayTransition) -> ActionValidation:
    action = transition.action
    if not isinstance(action, list) or any(not isinstance(index, int) or isinstance(index, bool) for index in action):
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


def _numeric_reward(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    reward = float(value)
    return reward if math.isfinite(reward) else None


def _terminal_reward_mapping(steps: list[Any]) -> tuple[bool, dict[int, float] | None]:
    """Return whether terminal rewards exist and their validated player mapping.

    ``(False, None)`` means no terminal ``reward`` field was supplied, so legacy
    ``observation.current.result`` fallback is allowed. ``(True, None)`` means a
    terminal reward was supplied but was incomplete or malformed; callers must not
    fall back and silently override that ambiguous evidence.
    """

    for step in reversed(steps):
        if not isinstance(step, list):
            continue
        done_views = [
            (player, view)
            for player, view in enumerate(step)
            if isinstance(view, dict) and view.get("status") == "DONE"
        ]
        if not done_views or not any("reward" in view for _, view in done_views):
            continue
        if len(done_views) < 2:
            return True, None
        rewards: dict[int, float] = {}
        for player, view in done_views:
            reward = _numeric_reward(view.get("reward"))
            if reward is None:
                return True, None
            rewards[player] = reward
        return True, rewards
    return False, None


def _top_level_reward_mapping(replay: dict[str, Any]) -> tuple[bool, dict[int, float] | None]:
    if "rewards" not in replay:
        return False, None
    values = replay.get("rewards")
    if not isinstance(values, list) or len(values) < 2:
        return True, None
    rewards: dict[int, float] = {}
    for player, value in enumerate(values):
        reward = _numeric_reward(value)
        if reward is None:
            return True, None
        rewards[player] = reward
    return True, rewards


def _winner_from_rewards(rewards: dict[int, float] | None) -> int | None:
    if not rewards or len(rewards) < 2:
        return None
    winners = [player for player, reward in rewards.items() if reward > 0]
    losers = [player for player, reward in rewards.items() if reward < 0]
    if len(winners) == 1 and len(losers) == len(rewards) - 1:
        return winners[0]
    if all(reward == 0 for reward in rewards.values()):
        return 2
    return None


def _reward_mappings_match(left: dict[int, float] | None, right: dict[int, float] | None) -> bool:
    if left is None or right is None or left.keys() != right.keys():
        return False
    return all(math.isclose(left[player], right[player], rel_tol=0.0, abs_tol=1e-12) for player in left)


def _terminal_forfeit_winner(replay: dict[str, Any], steps: list[Any]) -> int | None:
    """Resolve a Kaggle timeout/error forfeit from matching terminal evidence.

    Kaggle records these episodes with one ``DONE`` player carrying a positive
    reward and one ``TIMEOUT`` or ``ERROR`` player carrying ``null``.  Accept the
    result only when the final per-player views and the top-level status/reward
    vectors agree exactly; otherwise the ordinary incomplete-reward gate remains
    blocking.
    """

    top_statuses = replay.get("statuses")
    top_rewards = replay.get("rewards")
    if not isinstance(top_statuses, list) or not isinstance(top_rewards, list):
        return None
    if len(top_statuses) != 2 or len(top_rewards) != 2:
        return None

    for step in reversed(steps):
        if not isinstance(step, list) or len(step) != 2 or not all(isinstance(view, dict) for view in step):
            continue
        step_statuses = [view.get("status") for view in step]
        step_rewards = [view.get("reward") for view in step]
        if step_statuses != top_statuses or step_rewards != top_rewards:
            continue
        winners = [
            player
            for player, (status, reward) in enumerate(zip(step_statuses, step_rewards))
            if status == "DONE" and (_numeric_reward(reward) or 0.0) > 0.0
        ]
        forfeits = [
            player
            for player, (status, reward) in enumerate(zip(step_statuses, step_rewards))
            if status in {"TIMEOUT", "ERROR"} and reward is None
        ]
        if len(winners) == 1 and len(forfeits) == 1 and winners[0] != forfeits[0]:
            return winners[0]
    return None


def terminal_outcome(replay: dict[str, Any]) -> dict[str, Any]:
    """Resolve terminal outcome and retain auditable reward-source metadata."""

    steps = replay.get("steps") or []
    terminal_reward_present, terminal_rewards = _terminal_reward_mapping(steps)
    top_level_reward_present, top_level_rewards = _top_level_reward_mapping(replay)
    forfeit_winner = _terminal_forfeit_winner(replay, steps)
    if terminal_reward_present and terminal_rewards is None and forfeit_winner is not None:
        return {
            "winner": forfeit_winner,
            "winner_source": "terminal_forfeit",
            "terminal_reward_present": True,
            "terminal_reward_valid": False,
            "top_level_reward_present": top_level_reward_present,
            "top_level_reward_valid": False,
            "reward_mismatch": False,
        }
    reward_mismatch = (
        terminal_reward_present
        and top_level_reward_present
        and not _reward_mappings_match(terminal_rewards, top_level_rewards)
    )

    if terminal_reward_present:
        winner = _winner_from_rewards(terminal_rewards)
        return {
            "winner": winner,
            "winner_source": "terminal_reward" if winner is not None else "unresolved_terminal_reward",
            "terminal_reward_present": True,
            "terminal_reward_valid": terminal_rewards is not None,
            "top_level_reward_present": top_level_reward_present,
            "top_level_reward_valid": top_level_rewards is not None,
            "reward_mismatch": reward_mismatch,
        }

    for step in reversed(steps):
        if not isinstance(step, list):
            continue
        for view in step:
            if not isinstance(view, dict):
                continue
            observation = view.get("observation") or {}
            current = observation.get("current") or {}
            result = current.get("result")
            if isinstance(result, int) and not isinstance(result, bool) and result != -1:
                return {
                    "winner": result,
                    "winner_source": "observation_result",
                    "terminal_reward_present": False,
                    "terminal_reward_valid": False,
                    "top_level_reward_present": top_level_reward_present,
                    "top_level_reward_valid": top_level_rewards is not None,
                    "reward_mismatch": False,
                }
    return {
        "winner": None,
        "winner_source": "unresolved",
        "terminal_reward_present": False,
        "terminal_reward_valid": False,
        "top_level_reward_present": top_level_reward_present,
        "top_level_reward_valid": top_level_rewards is not None,
        "reward_mismatch": False,
    }


def terminal_winner(replay: dict[str, Any]) -> int | None:
    return terminal_outcome(replay)["winner"]


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
    for transition in iter_transitions(replay, alignment, acting_only=False):
        counts["recorded_actions"] += 1
        if transition.action_step == 0:
            counts["initial_actions_skipped"] += 1
            continue
        if transition.submission_status is None:
            counts["unknown_submission_status_skipped"] += 1
            continue
        if transition.submission_status != "ACTIVE":
            counts["non_acting_actions_skipped"] += 1
            continue
        counts["actions"] += 1
        validation = validate_transition(transition)
        counts[validation.kind] += 1
        reasons[validation.reason] += 1
        if transition.action == []:
            counts["empty_actions"] += 1
            counts["valid_empty_decisions" if validation.valid else "invalid_empty_decisions"] += 1
        if not validation.valid and len(examples) < max_examples:
            examples.append(
                {
                    "episode_id": transition.episode_id,
                    "player": transition.player,
                    "action_step": transition.action_step,
                    "observation_step": transition.observation_step,
                    "action": transition.action,
                    "submission_status": transition.submission_status,
                    "action_status": transition.action_status,
                    "observation_status": transition.observation_status,
                    **asdict(validation),
                }
            )
    decisions = counts["decision"] + counts["invalid"]
    return {
        "alignment": alignment,
        "status_source": "steps[action_step-1].status",
        "recorded_actions": counts["recorded_actions"],
        "actions": counts["actions"],
        "initial_actions_skipped": counts["initial_actions_skipped"],
        "non_acting_actions_skipped": counts["non_acting_actions_skipped"],
        "unknown_submission_status_skipped": counts["unknown_submission_status_skipped"],
        "empty_actions": counts["empty_actions"],
        "valid_empty_decisions": counts["valid_empty_decisions"],
        "invalid_empty_decisions": counts["invalid_empty_decisions"],
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
        for key in (
            "recorded_actions",
            "actions",
            "initial_actions_skipped",
            "non_acting_actions_skipped",
            "unknown_submission_status_skipped",
            "empty_actions",
            "valid_empty_decisions",
            "invalid_empty_decisions",
            "valid_decisions",
            "setup_actions",
            "invalid_decisions",
        ):
            totals[key] += int(report.get(key, 0))
        reasons.update(report.get("reasons") or {})
        examples.extend((report.get("examples") or [])[: max_examples - len(examples)])
    decisions = totals["valid_decisions"] + totals["invalid_decisions"]
    return {
        "alignment": alignment,
        "status_source": "steps[action_step-1].status",
        **{
            key: totals[key]
            for key in (
                "recorded_actions",
                "actions",
                "initial_actions_skipped",
                "non_acting_actions_skipped",
                "unknown_submission_status_skipped",
                "empty_actions",
                "valid_empty_decisions",
                "invalid_empty_decisions",
                "valid_decisions",
                "setup_actions",
                "invalid_decisions",
            )
        },
        "valid_rate": totals["valid_decisions"] / decisions if decisions else 0.0,
        "reasons": dict(sorted(reasons.items())),
        "examples": examples,
    }


def _select_summary(observation: dict[str, Any] | None) -> dict[str, Any]:
    select = None if observation is None else observation.get("select")
    if not isinstance(select, dict):
        return {"present": False}
    options = select.get("option")
    option_list = options if isinstance(options, list) else []
    option_types = Counter(
        str(option.get("type"))
        for option in option_list
        if isinstance(option, dict) and option.get("type") is not None
    )
    fingerprint = hashlib.sha256(json_dumps(select).encode("utf-8")).hexdigest()[:16]
    return {
        "present": True,
        "fingerprint": fingerprint,
        "type": select.get("type"),
        "context": select.get("context"),
        "option_count": len(option_list),
        "option_types": dict(sorted(option_types.items())),
        "min_count": select.get("minCount"),
        "max_count": select.get("maxCount"),
    }


def _empty_required_selection(transition: ReplayTransition) -> bool:
    if transition.action != [] or transition.observation is None:
        return False
    select = transition.observation.get("select")
    if not isinstance(select, dict):
        return False
    try:
        return int(select.get("minCount", 0)) > 0
    except (TypeError, ValueError):
        return False


def audit_action_positions(
    replay: dict[str, Any],
    *,
    lags: tuple[int, ...] = (-1, 0, 1, 2, 3, 4),
    expected_lag: int = 1,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Compare where actions appear relative to ACTIVE observations.

    ``lag`` is ``action_step - observation_step``.  Kaggle Environments' core
    execution model predicts lag 1.  Other lags are reported only to diagnose a
    competition-specific wrapper or repeated/auto-resolved selection; they are not
    automatically accepted for DATA-001 or DATA-002.
    """

    if expected_lag not in lags:
        raise ValueError("expected_lag must be included in lags")
    steps = replay.get("steps") or []
    episode_id = replay_episode_id(replay)
    lag_counts: dict[int, Counter[str]] = {lag: Counter() for lag in lags}
    lag_reasons: dict[int, Counter[str]] = {lag: Counter() for lag in lags}
    totals: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for observation_step, step in enumerate(steps):
        if not isinstance(step, list):
            continue
        for player, observation_view in enumerate(step):
            if not isinstance(observation_view, dict) or observation_view.get("status") != "ACTIVE":
                continue
            observation = observation_view.get("observation")
            if not isinstance(observation, dict):
                totals["active_missing_observation"] += 1
                continue
            totals["active_observations"] += 1
            candidates: list[dict[str, Any]] = []
            expected_transition: ReplayTransition | None = None
            expected_validation: ActionValidation | None = None
            valid_other_lags: list[int] = []

            for lag in lags:
                action_step = observation_step + lag
                transition = _transition_at(
                    steps,
                    episode_id=episode_id,
                    player=player,
                    action_step=action_step,
                    observation_step=observation_step,
                )
                validation = validate_transition(transition)
                counter = lag_counts[lag]
                counter["observations"] += 1
                counter[validation.kind] += 1
                lag_reasons[lag][validation.reason] += 1
                if transition.action is None:
                    counter["missing_actions"] += 1
                elif transition.action == []:
                    counter["empty_actions"] += 1
                if lag == expected_lag:
                    expected_transition = transition
                    expected_validation = validation
                elif validation.valid:
                    valid_other_lags.append(lag)
                candidates.append(
                    {
                        "lag": lag,
                        "action_step": action_step,
                        "action": transition.action,
                        "submission_status": transition.submission_status,
                        "action_status": transition.action_status,
                        "valid": validation.valid,
                        "kind": validation.kind,
                        "reason": validation.reason,
                    }
                )

            assert expected_transition is not None and expected_validation is not None
            if expected_transition.action == []:
                totals["empty_expected_actions"] += 1
            if _empty_required_selection(expected_transition):
                totals["empty_required_expected_actions"] += 1
                if valid_other_lags:
                    totals["empty_required_with_valid_other_lag"] += 1
                    for lag in valid_other_lags:
                        totals[f"valid_other_lag_{lag}"] += 1
                else:
                    totals["empty_required_unresolved"] += 1
                if len(examples) < max_examples:
                    examples.append(
                        {
                            "episode_id": episode_id,
                            "player": player,
                            "observation_step": observation_step,
                            "observation_status": observation_view.get("status"),
                            "select": _select_summary(observation),
                            "expected_lag": expected_lag,
                            "expected_reason": expected_validation.reason,
                            "valid_other_lags": valid_other_lags,
                            "candidates": candidates,
                        }
                    )

    lag_reports: dict[str, dict[str, Any]] = {}
    for lag in lags:
        counter = lag_counts[lag]
        decisions = counter["decision"] + counter["invalid"]
        lag_reports[str(lag)] = {
            "lag": lag,
            "observations": counter["observations"],
            "valid_decisions": counter["decision"],
            "setup_actions": counter["setup"],
            "invalid_decisions": counter["invalid"],
            "valid_rate": counter["decision"] / decisions if decisions else 0.0,
            "empty_actions": counter["empty_actions"],
            "missing_actions": counter["missing_actions"],
            "reasons": dict(sorted(lag_reasons[lag].items())),
        }
    return {
        "episode_id": episode_id,
        "expected_lag": expected_lag,
        "active_observations": totals["active_observations"],
        "active_missing_observation": totals["active_missing_observation"],
        "empty_expected_actions": totals["empty_expected_actions"],
        "empty_required_expected_actions": totals["empty_required_expected_actions"],
        "empty_required_with_valid_other_lag": totals["empty_required_with_valid_other_lag"],
        "empty_required_unresolved": totals["empty_required_unresolved"],
        "valid_other_lags": {
            str(lag): totals[f"valid_other_lag_{lag}"] for lag in lags if lag != expected_lag
        },
        "lags": lag_reports,
        "examples": examples,
    }


def merge_position_audits(
    reports: list[dict[str, Any]],
    *,
    lags: tuple[int, ...],
    expected_lag: int,
    max_examples: int = 50,
) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    lag_totals: dict[int, Counter[str]] = {lag: Counter() for lag in lags}
    lag_reasons: dict[int, Counter[str]] = {lag: Counter() for lag in lags}
    other_lags: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for report in reports:
        for key in (
            "active_observations",
            "active_missing_observation",
            "empty_expected_actions",
            "empty_required_expected_actions",
            "empty_required_with_valid_other_lag",
            "empty_required_unresolved",
        ):
            totals[key] += int(report.get(key, 0))
        other_lags.update(report.get("valid_other_lags") or {})
        examples.extend((report.get("examples") or [])[: max_examples - len(examples)])
        for lag in lags:
            lag_report = (report.get("lags") or {}).get(str(lag), {})
            for key in (
                "observations",
                "valid_decisions",
                "setup_actions",
                "invalid_decisions",
                "empty_actions",
                "missing_actions",
            ):
                lag_totals[lag][key] += int(lag_report.get(key, 0))
            lag_reasons[lag].update(lag_report.get("reasons") or {})

    lag_reports: dict[str, dict[str, Any]] = {}
    for lag in lags:
        counter = lag_totals[lag]
        decisions = counter["valid_decisions"] + counter["invalid_decisions"]
        lag_reports[str(lag)] = {
            "lag": lag,
            **{key: counter[key] for key in (
                "observations",
                "valid_decisions",
                "setup_actions",
                "invalid_decisions",
                "empty_actions",
                "missing_actions",
            )},
            "valid_rate": counter["valid_decisions"] / decisions if decisions else 0.0,
            "reasons": dict(sorted(lag_reasons[lag].items())),
        }
    best_lag = max(lags, key=lambda lag: (lag_reports[str(lag)]["valid_rate"], lag_reports[str(lag)]["valid_decisions"])) if reports else None
    return {
        "expected_lag": expected_lag,
        "best_observed_lag": best_lag,
        **dict(totals),
        "valid_other_lags": dict(sorted(other_lags.items(), key=lambda item: int(item[0]))),
        "lags": lag_reports,
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
    terminal = terminal_outcome(replay)
    winner = terminal["winner"]
    rows: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    setup_actions = 0
    episode_id = replay_episode_id(replay, Path(source_path).stem)
    skipped: Counter[str] = Counter()
    for transition in iter_transitions(replay, alignment, acting_only=False):
        if transition.action_step == 0:
            skipped["initial_actions_skipped"] += 1
            continue
        if transition.submission_status is None:
            skipped["unknown_submission_status_skipped"] += 1
            continue
        if transition.submission_status != "ACTIVE":
            skipped["non_acting_actions_skipped"] += 1
            continue
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
                    "submission_status": transition.submission_status,
                    "action_status": transition.action_status,
                    "observation_status": transition.observation_status,
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
                "schema_version": 2,
                "episode_id": episode_id,
                "source_path": source_path,
                "source_sha256": source_sha256,
                "player": transition.player,
                "action_step": transition.action_step,
                "observation_step": transition.observation_step,
                "status": transition.submission_status,
                "submission_status": transition.submission_status,
                "observation_status": transition.observation_status,
                "action_status": transition.action_status,
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
        **terminal,
        "rows": len(rows),
        "setup_actions": setup_actions,
        **dict(skipped),
        "invalid_decisions": len(invalid),
        "invalid_examples": invalid[:20],
    }


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
