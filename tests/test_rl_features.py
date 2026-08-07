from __future__ import annotations

import unittest

from rl.features import ACTION_DIM, STATE_DIM, action_features, state_features
from scripts.train_rl_policy import TrajectoryDataset, split_by_episode


class RLFeatureTests(unittest.TestCase):
    def test_empty_observation_has_stable_shape(self) -> None:
        self.assertEqual(len(state_features({})), STATE_DIM)

    def test_current_player_is_canonicalized(self) -> None:
        player = {"deckCount": 40, "prize": [None] * 5, "handCount": 7, "benchMax": 5}
        observation = {"current": {"yourIndex": 1, "players": [{}, player]}, "select": {"type": 2}}
        vector = state_features(observation)
        self.assertAlmostEqual(vector[8], 40 / 60)

    def test_option_shape(self) -> None:
        vector = action_features({"type": 3, "cardId": 1172}, 4)
        self.assertEqual(len(vector), ACTION_DIM)
        self.assertEqual(vector[-3], 1.0)

    def test_episode_split_has_no_leakage(self) -> None:
        rows = [
            {"episode": episode, "chosen": [0], "options": [[0.0] * ACTION_DIM]}
            for episode in range(10)
            for _ in range(2)
        ]
        train, validation = split_by_episode(TrajectoryDataset(rows), 0.2, 7)
        train_episodes = {row["episode"] for row in train.rows}
        validation_episodes = {row["episode"] for row in validation.rows}
        self.assertFalse(train_episodes & validation_episodes)
        self.assertEqual(len(validation_episodes), 2)


if __name__ == "__main__":
    unittest.main()
