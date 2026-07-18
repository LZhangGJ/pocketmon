from __future__ import annotations

import unittest

from rl.features import ACTION_DIM, STATE_DIM, action_features, state_features


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


if __name__ == "__main__":
    unittest.main()
