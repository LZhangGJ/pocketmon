from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import torch

from rl.agent_adapter import RLBCPolicyAdapter, STRUCTURED_ARCHITECTURE
from rl.bc import batch_loss, collate_rows, greedy_decode, load_deck_map
from rl.features import STATE_DIM, action_features, state_features, structured_observation_features
from rl.model import StructuredMaskedPointerActorCritic


def observation() -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "turn": 3,
            "players": [
                {
                    "active": [{"id": 10, "hp": 100, "maxHp": 120, "energies": [1], "tools": []}],
                    "bench": [{"id": 11, "hp": 60, "maxHp": 60}],
                    "hand": [{"id": 12}], "discard": [{"id": 13}], "prize": [None] * 6,
                },
                {
                    "active": [{"id": 20, "hp": 80, "maxHp": 100}], "bench": [],
                    "hand": [{"id": 999}], "discard": [{"id": 21}], "prize": [{"id": 998}],
                },
            ],
            "stadium": [{"id": 30}],
        },
        "select": {
            "type": 0, "minCount": 1, "maxCount": 1,
            "contextCard": {"id": 31},
            "option": [
                {"type": 7, "index": 0},
                {"type": 13, "attackId": 44},
                {"type": 3, "area": 5, "index": 0, "playerIndex": 0},
            ],
        },
    }


def compact_structured_row() -> dict:
    obs = observation()
    options = obs["select"]["option"]
    row = {
        "state": state_features(obs),
        "options": [action_features(value, index) for index, value in enumerate(options)],
        "action": [0], "history": [], "min_count": 1, "max_count": 1,
        "outcome": 1.0, "policy_weight": 1.0, "value_weight": 1.0,
    }
    row.update(structured_observation_features(obs, options))
    row["deck_card_ids"] = [10, 11, 12] * 20
    return row


class StructuredRLTests(unittest.TestCase):
    def test_visible_entities_exclude_hidden_opponent_hand_and_prize(self) -> None:
        value = structured_observation_features(observation(), observation()["select"]["option"])
        self.assertNotIn(999, value["entity_card_ids"])
        self.assertNotIn(998, value["entity_card_ids"])
        self.assertTrue({10, 11, 12, 13, 20, 21, 30, 31}.issubset(value["entity_card_ids"]))
        self.assertEqual(value["option_card_ids"][0], 12)
        self.assertEqual(value["option_attack_ids"][1], 44)
        self.assertEqual(value["option_card_ids"][2], 11)

    def test_structured_model_loss_and_decode(self) -> None:
        row = compact_structured_row()
        batch = collate_rows([row, row])
        model = StructuredMaskedPointerActorCritic(32)
        loss, _ = batch_loss(model, batch)
        self.assertTrue(torch.isfinite(loss))
        predictions = greedy_decode(model.eval(), batch)
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(len(action) == 1 for action in predictions))
        self.assertEqual(tuple(batch["deck_card_ids"].shape), (2, 60))

    def test_deck_map_requires_own_60_card_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "decks.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for player in (0, 1):
                    handle.write(json.dumps({"episode_id": "e", "player": player, "deck": [player + 1] * 60}) + "\n")
            decks, audit = load_deck_map(path)
        self.assertEqual(decks[("e", 0)], [1] * 60)
        self.assertEqual(decks[("e", 1)], [2] * 60)
        self.assertEqual(audit["entries"], 2)

    def test_confidence_gate_uses_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structured.pt"
            model = StructuredMaskedPointerActorCritic(32)
            torch.save({
                "model": model.state_dict(), "hidden_dim": 32,
                "config": {"architecture": STRUCTURED_ARCHITECTURE, "confidence_threshold": 1.0},
            }, path)
            adapter = RLBCPolicyAdapter(path, fallback=lambda _: [2])
            action = adapter.act(observation())
        self.assertEqual(action, [2])
        self.assertEqual(adapter.diagnostics()["low_confidence_actions"], 1)
        self.assertEqual(adapter.diagnostics()["fallback_actions"], 1)


if __name__ == "__main__":
    unittest.main()
