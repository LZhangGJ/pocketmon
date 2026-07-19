from __future__ import annotations

import unittest

from rl.public_replay import (
    audit_action_positions,
    audit_replay,
    canonical_rows,
    iter_transitions,
    terminal_winner,
    validate_transition,
)


def observation(option_count: int | None, result: int = -1, logs: bool = True) -> dict:
    payload = {"current": {"result": result, "yourIndex": 0, "players": [{}, {}]}}
    if option_count is not None:
        payload["select"] = {
            "type": 1,
            "context": 7,
            "option": [{"type": 3, "cardId": 100 + index} for index in range(option_count)],
            "minCount": 1,
            "maxCount": 1,
        }
    if logs:
        payload["logs"] = ["must not become a model feature"]
    return payload


def shifted_replay() -> dict:
    # Each action is stored on the following step beside the post-action observation.
    return {
        "info": {"EpisodeId": 42},
        "steps": [
            [
                {"action": None, "status": "ACTIVE", "observation": observation(3)},
                {"action": None, "status": "ACTIVE", "observation": observation(2)},
            ],
            [
                {"action": [2], "status": "ACTIVE", "observation": observation(1)},
                {"action": [1], "status": "ACTIVE", "observation": observation(4)},
            ],
            [
                {"action": [0], "status": "DONE", "observation": observation(None, result=0)},
                {"action": [3], "status": "DONE", "observation": observation(None, result=0)},
            ],
        ],
    }


def turn_switch_replay() -> dict:
    """The action-step status is post-action and may be the opposite player's turn."""

    return {
        "info": {"EpisodeId": 43},
        "steps": [
            [
                {"action": None, "status": "ACTIVE", "observation": observation(2)},
                {"action": None, "status": "INACTIVE", "observation": observation(2)},
            ],
            [
                {"action": [1], "status": "INACTIVE", "observation": observation(1)},
                {"action": [], "status": "ACTIVE", "observation": observation(3)},
            ],
            [
                {"action": [], "status": "DONE", "observation": observation(None, result=1)},
                {"action": [2], "status": "DONE", "observation": observation(None, result=1)},
            ],
        ],
    }


def delayed_action_replay() -> dict:
    return {
        "info": {"EpisodeId": 44},
        "steps": [
            [{"action": None, "status": "ACTIVE", "observation": observation(2)}],
            [{"action": [], "status": "ACTIVE", "observation": observation(2)}],
            [{"action": [1], "status": "DONE", "observation": observation(None, result=0)}],
        ],
    }


def reward_only_replay(rewards: tuple[float, float]) -> dict:
    """Mimics real replays: `observation.current.result` never leaves -1, but the
    terminal step's `reward` resolves the winner."""

    return {
        "info": {"EpisodeId": 45},
        "rewards": list(rewards),
        "steps": [
            [
                {"action": None, "status": "ACTIVE", "observation": observation(2), "reward": 0},
                {"action": None, "status": "ACTIVE", "observation": observation(2), "reward": 0},
            ],
            [
                {
                    "action": [0],
                    "status": "DONE",
                    "observation": observation(None, result=-1),
                    "reward": rewards[0],
                },
                {
                    "action": [1],
                    "status": "DONE",
                    "observation": observation(None, result=-1),
                    "reward": rewards[1],
                },
            ],
        ],
    }


class PublicReplayTests(unittest.TestCase):
    def test_previous_alignment_is_valid_and_same_alignment_is_not(self) -> None:
        replay = shifted_replay()
        previous = audit_replay(replay, "previous")
        same = audit_replay(replay, "same")
        self.assertEqual(previous["valid_decisions"], 4)
        self.assertEqual(previous["invalid_decisions"], 0)
        self.assertGreater(same["invalid_decisions"], 0)

    def test_validator_rejects_out_of_range_index(self) -> None:
        transition = next(iter_transitions(shifted_replay(), "same"))
        validation = validate_transition(transition)
        self.assertFalse(validation.valid)
        self.assertEqual(validation.reason, "option_index_out_of_range")

    def test_submission_status_comes_from_previous_step(self) -> None:
        replay = turn_switch_replay()
        transitions = list(iter_transitions(replay, "previous"))
        self.assertEqual([(row.player, row.action_step) for row in transitions], [(0, 1), (1, 2)])
        self.assertEqual(transitions[0].submission_status, "ACTIVE")
        self.assertEqual(transitions[0].action_status, "INACTIVE")
        audit = audit_replay(replay, "previous")
        self.assertEqual(audit["valid_decisions"], 2)
        self.assertEqual(audit["invalid_decisions"], 0)
        self.assertEqual(audit["non_acting_actions_skipped"], 2)
        rows, conversion = canonical_rows(
            replay,
            alignment="previous",
            source_path="43.json",
            source_sha256="def",
            policy_source="all",
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(conversion["invalid_decisions"], 0)
        self.assertEqual(conversion["non_acting_actions_skipped"], 2)

    def test_action_position_audit_finds_later_submission(self) -> None:
        report = audit_action_positions(
            delayed_action_replay(),
            lags=(0, 1, 2),
            expected_lag=1,
        )
        self.assertEqual(report["empty_required_expected_actions"], 1)
        self.assertEqual(report["empty_required_with_valid_other_lag"], 1)
        self.assertEqual(report["valid_other_lags"]["2"], 1)

    def test_validator_rejects_boolean_option_index(self) -> None:
        replay = shifted_replay()
        replay["steps"][1][0]["action"] = [True]
        transition = next(iter_transitions(replay, "previous"))
        validation = validate_transition(transition)
        self.assertFalse(validation.valid)
        self.assertEqual(validation.reason, "action_not_integer_list")

    def test_canonical_rows_use_one_player_view_and_strip_logs(self) -> None:
        rows, report = canonical_rows(
            shifted_replay(),
            alignment="previous",
            source_path="42.json",
            source_sha256="abc",
            manifest={"rating": "1200"},
        )
        self.assertEqual(report["invalid_decisions"], 0)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]["schema_version"], 2)
        self.assertEqual(rows[0]["submission_status"], "ACTIVE")
        self.assertNotIn("logs", rows[0]["observation"])
        self.assertEqual(rows[0]["chosen_options"], [{"type": 3, "cardId": 102}])
        winner_rows = [row for row in rows if row["player"] == 0]
        loser_rows = [row for row in rows if row["player"] == 1]
        self.assertTrue(all(row["policy_weight"] == 1.0 for row in winner_rows))
        self.assertTrue(all(row["policy_weight"] == 0.0 for row in loser_rows))
        self.assertTrue(all(row["value_weight"] == 1.0 for row in rows))

    def test_terminal_winner_uses_reward_when_result_stays_negative_one(self) -> None:
        self.assertEqual(terminal_winner(reward_only_replay((1, -1))), 0)
        self.assertEqual(terminal_winner(reward_only_replay((-1, 1))), 1)

    def test_terminal_winner_draw_from_zero_rewards(self) -> None:
        self.assertEqual(terminal_winner(reward_only_replay((0, 0))), 2)

    def test_terminal_winner_falls_back_to_result_when_no_reward(self) -> None:
        # Existing synthetic fixtures carry no "reward" key at all.
        self.assertEqual(terminal_winner(shifted_replay()), 0)

    def test_terminal_winner_none_for_unrecognized_reward_pattern(self) -> None:
        self.assertIsNone(terminal_winner(reward_only_replay((1, 1))))


if __name__ == "__main__":
    unittest.main()
