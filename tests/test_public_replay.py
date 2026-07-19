from __future__ import annotations

import unittest

from rl.public_replay import audit_replay, canonical_rows, iter_transitions, validate_transition


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
                {"action": None, "observation": observation(3)},
                {"action": None, "observation": observation(2)},
            ],
            [
                {"action": [2], "observation": observation(1)},
                {"action": [1], "observation": observation(4)},
            ],
            [
                {"action": [0], "observation": observation(None, result=0)},
                {"action": [3], "observation": observation(None, result=0)},
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
        self.assertNotIn("logs", rows[0]["observation"])
        self.assertEqual(rows[0]["chosen_options"], [{"type": 3, "cardId": 102}])
        winner_rows = [row for row in rows if row["player"] == 0]
        loser_rows = [row for row in rows if row["player"] == 1]
        self.assertTrue(all(row["policy_weight"] == 1.0 for row in winner_rows))
        self.assertTrue(all(row["policy_weight"] == 0.0 for row in loser_rows))
        self.assertTrue(all(row["value_weight"] == 1.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
