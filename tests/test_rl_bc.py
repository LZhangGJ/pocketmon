from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from rl.bc import (
    FORBIDDEN_MODEL_FIELDS,
    MODEL_INPUT_FIELDS,
    TrajectoryDataset,
    action_is_legal,
    batch_loss,
    collate_rows,
    compact_replay_row,
    greedy_decode,
    split_by_episode,
)
from rl.features import ACTION_DIM, STATE_DIM
from rl.model import MaskedPointerActorCritic, legal_choice_mask


def compact(action: list[int], min_count: int, max_count: int, policy_weight: float = 1.0, episode: str = "1") -> dict:
    return {
        "episode_id": episode,
        "state": [0.0] * STATE_DIM,
        "options": [[0.0] * ACTION_DIM for _ in range(4)],
        "action": action,
        "min_count": min_count,
        "max_count": max_count,
        "outcome": 1.0 if policy_weight else -1.0,
        "policy_weight": policy_weight,
        "value_weight": 1.0,
        "select_type": 1,
        "select_context": "0",
        "source_sha256": "abc",
    }


class RLBehaviorCloningTests(unittest.TestCase):
    def test_episode_split_has_no_leakage(self) -> None:
        rows = [compact([0], 1, 1, episode=str(episode)) for episode in range(20) for _ in range(2)]
        train, validation = split_by_episode(TrajectoryDataset(rows), 0.2, 20260720)
        train_ids = {row["episode_id"] for row in train.rows}
        validation_ids = {row["episode_id"] for row in validation.rows}
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(len(validation_ids), 4)
        self.assertEqual(len(train) + len(validation), len(rows))

    def test_empty_action_is_legal_when_optional(self) -> None:
        self.assertTrue(action_is_legal([], 3, 0, 2))
        raw = {
            "schema_version": 2, "episode_id": 1, "action": [], "outcome": 1.0,
            "policy_weight": 1.0, "value_weight": 1.0, "source_sha256": "x",
            "observation": {"current": {}, "select": {"option": [{}, {}], "minCount": 0, "maxCount": 2}},
        }
        self.assertEqual(compact_replay_row(raw)["action"], [])

    def test_stop_masked_before_min_count(self) -> None:
        mask = legal_choice_mask(torch.tensor([[True, True]]), torch.tensor([[False, False]]), torch.tensor([0]), torch.tensor([1]), torch.tensor([2]))
        self.assertFalse(mask[0, -1])

    def test_max_count_forces_stop(self) -> None:
        mask = legal_choice_mask(torch.tensor([[True, True]]), torch.tensor([[True, False]]), torch.tensor([1]), torch.tensor([0]), torch.tensor([1]))
        self.assertTrue(mask[0, -1])
        self.assertFalse(mask[0, :-1].any())

    def test_selected_candidate_is_masked(self) -> None:
        mask = legal_choice_mask(torch.tensor([[True, True]]), torch.tensor([[True, False]]), torch.tensor([1]), torch.tensor([1]), torch.tensor([2]))
        self.assertFalse(mask[0, 0]); self.assertTrue(mask[0, 1])

    def test_single_select_teacher_forcing(self) -> None:
        model = MaskedPointerActorCritic(16)
        _, parts = batch_loss(model, collate_rows([compact([2], 1, 1)]))
        self.assertEqual(parts["policy_count"], 2.0)  # candidate plus STOP

    def test_multi_select_teacher_forcing_preserves_order(self) -> None:
        model = MaskedPointerActorCritic(16)
        _, parts = batch_loss(model, collate_rows([compact([2, 0], 2, 2)]))
        self.assertEqual(parts["policy_count"], 3.0)

    def test_mixed_action_lengths_do_not_create_nan(self) -> None:
        model = MaskedPointerActorCritic(16)
        loss, _ = batch_loss(model, collate_rows([compact([], 0, 2), compact([2, 0], 2, 2)]))
        self.assertTrue(torch.isfinite(loss))

    def test_greedy_decode_is_always_legal(self) -> None:
        torch.manual_seed(3)
        model = MaskedPointerActorCritic(16)
        rows = [compact([], 0, 3), compact([1], 1, 1), compact([0, 2], 2, 3)]
        batch = collate_rows(rows)
        predictions = greedy_decode(model, batch)
        for row, prediction in zip(rows, predictions):
            self.assertTrue(action_is_legal(prediction, len(row["options"]), row["min_count"], row["max_count"]))

    def test_loser_policy_loss_weight_is_zero(self) -> None:
        model = MaskedPointerActorCritic(16)
        loss, parts = batch_loss(model, collate_rows([compact([0], 1, 1, policy_weight=0.0)]))
        self.assertEqual(parts["policy_count"], 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_value_head_uses_both_players(self) -> None:
        model = MaskedPointerActorCritic(16)
        _, parts = batch_loss(model, collate_rows([compact([0], 1, 1, 1.0), compact([1], 1, 1, 0.0)]))
        self.assertEqual(parts["value_count"], 2.0)

    def test_model_inputs_exclude_labels_and_logs(self) -> None:
        self.assertFalse(MODEL_INPUT_FIELDS & FORBIDDEN_MODEL_FIELDS)
        self.assertNotIn("winner", MODEL_INPUT_FIELDS)
        self.assertNotIn("outcome", MODEL_INPUT_FIELDS)
        self.assertNotIn("action_status", MODEL_INPUT_FIELDS)

    def test_checkpoint_reload_has_identical_output(self) -> None:
        torch.manual_seed(9)
        model = MaskedPointerActorCritic(16).eval()
        states = torch.randn(2, STATE_DIM)
        options = torch.randn(2, 3, ACTION_DIM)
        before = model(states, options)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"model": model.state_dict()}, path)
            restored = MaskedPointerActorCritic(16).eval()
            restored.load_state_dict(torch.load(path, weights_only=False)["model"])
        after = restored(states, options)
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertTrue(torch.equal(before[1], after[1]))


if __name__ == "__main__":
    unittest.main()
