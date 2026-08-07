from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rl.bc import (
    FORBIDDEN_MODEL_FIELDS,
    HISTORY_MODEL_INPUT_FIELDS,
    MODEL_INPUT_FIELDS,
    TrajectoryDataset,
    action_is_legal,
    apply_replay_bias_correction,
    batch_loss,
    build_causal_histories,
    collate_rows,
    compact_replay_row,
    greedy_decode,
    split_by_episode,
)
from rl.features import ACTION_DIM, HISTORY_DIM, STATE_DIM, history_features
from rl.model import MaskedPointerActorCritic, legal_choice_mask
from scripts.evaluate_rl_checkpoint import validate_input_sha
from scripts.train_rl_policy import (
    ARCHITECTURE,
    HISTORY_ARCHITECTURE,
    achieved_seeds_for_fingerprint,
    actual_fingerprint_payload,
    assert_formal_worktree_clean,
    current_config_matches,
    experiment_fingerprint,
    new_training_state,
    restore_training_state,
    update_training_state,
    validate_resume_compatibility,
)
from scripts.summarize_rl_bc_experiment import aggregate_reports


def compact(action: list[int], min_count: int, max_count: int, policy_weight: float = 1.0, episode: str = "1") -> dict:
    return {
        "episode_id": episode,
        "player": 0,
        "action_step": 2,
        "observation_step": 1,
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

    def test_sample_weight_scales_policy_and_value_loss_mass(self) -> None:
        row = compact([0], 1, 1)
        row["sample_weight"] = 0.5
        _, parts = batch_loss(MaskedPointerActorCritic(16), collate_rows([row]))
        self.assertEqual(parts["policy_count"], 1.0)  # candidate plus STOP, each weighted 0.5
        self.assertEqual(parts["value_count"], 0.5)

    def test_replay_bias_correction_uses_recency_rating_and_both_decks(self) -> None:
        rows = []
        deck_map = {}
        metadata = (
            ("old-a", "2026-08-05T00:00:00Z", 1000.0, [1] * 60, [2] * 60),
            ("old-b", "2026-08-05T01:00:00Z", 1000.0, [1] * 60, [2] * 60),
            ("new", "2026-08-06T23:00:00Z", 1200.0, [3] * 60, [4] * 60),
        )
        for episode, created_at, rating, deck0, deck1 in metadata:
            for player in (0, 1):
                row = compact([0], 1, 1, episode=episode)
                row.update({"player": player, "created_at": created_at, "average_rating": rating})
                rows.append(row)
            deck_map[(episode, 0)] = deck0
            deck_map[(episode, 1)] = deck1
        audit = apply_replay_bias_correction(
            rows,
            deck_map,
            recency_half_life_days=2.0,
            deck_stratification_alpha=0.5,
            rating_stratification_alpha=0.5,
            min_sample_weight=0.1,
            max_sample_weight=10.0,
        )
        weights = {(row["episode_id"], row["player"]): row["sample_weight"] for row in rows}
        self.assertGreater(weights[("new", 0)], weights[("old-a", 0)])
        self.assertGreater(weights[("new", 1)], weights[("old-a", 1)])
        self.assertEqual(audit["own_deck_strata"], 4)
        self.assertEqual(audit["opponent_deck_strata"], 4)
        self.assertEqual(audit["rating_strata"], 2)
        self.assertEqual(audit["opponent_identity"], "submitted_deck_sha256_proxy")
        self.assertFalse(audit["agent_id_available"])

    def test_model_inputs_exclude_labels_and_logs(self) -> None:
        self.assertFalse(MODEL_INPUT_FIELDS & FORBIDDEN_MODEL_FIELDS)
        self.assertNotIn("winner", MODEL_INPUT_FIELDS)
        self.assertNotIn("outcome", MODEL_INPUT_FIELDS)
        self.assertNotIn("action_status", MODEL_INPUT_FIELDS)
        self.assertIn("sample_weight", FORBIDDEN_MODEL_FIELDS)

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

    def test_causal_history_uses_explicit_order_not_input_order(self) -> None:
        rows = []
        for step, action in ((6, [2]), (2, [0]), (4, [1])):
            row = compact(action, 1, 1)
            row.update({"action_step": step, "observation_step": step - 1})
            row["state"][0] = float(step)
            rows.append(row)
        prepared, audit = build_causal_histories(rows, max_history=2)
        by_step = {row["action_step"]: row for row in prepared}
        self.assertEqual(by_step[2]["history_steps"], [])
        self.assertEqual(by_step[4]["history_steps"], [2])
        self.assertEqual(by_step[6]["history_steps"], [2, 4])
        self.assertEqual(audit["physical_row_order_used"], False)
        self.assertEqual(audit["current_or_future_steps_used"], 0)

    def test_history_never_crosses_episode_or_player(self) -> None:
        rows = []
        for episode, player, step in (("a", 0, 2), ("a", 1, 3), ("a", 0, 4), ("b", 0, 5)):
            row = compact([0], 1, 1, episode=episode)
            row.update({"player": player, "action_step": step, "observation_step": step - 1})
            rows.append(row)
        prepared, audit = build_causal_histories(rows, max_history=8)
        lookup = {(row["episode_id"], row["player"], row["action_step"]): row for row in prepared}
        self.assertEqual(lookup[("a", 0, 4)]["history_steps"], [2])
        self.assertEqual(lookup[("a", 1, 3)]["history_steps"], [])
        self.assertEqual(lookup[("b", 0, 5)]["history_steps"], [])
        self.assertEqual(audit["groups"], 3)

    def test_history_token_excludes_current_action_future_and_outcome(self) -> None:
        previous = compact([1], 1, 1)
        previous.update({"action_step": 2, "observation_step": 1, "outcome": -99.0})
        current = compact([3], 1, 1)
        current.update({"action_step": 4, "observation_step": 3, "outcome": 99.0})
        future = compact([2], 1, 1)
        future.update({"action_step": 6, "observation_step": 5, "outcome": 123.0})
        prepared, _ = build_causal_histories([future, current, previous], max_history=4)
        current_prepared = next(row for row in prepared if row["action_step"] == 4)
        self.assertEqual(
            current_prepared["history"],
            [history_features(previous["state"], previous["options"], previous["action"])],
        )
        self.assertFalse(HISTORY_MODEL_INPUT_FIELDS & FORBIDDEN_MODEL_FIELDS)
        for forbidden in ("winner", "outcome", "action", "observation.logs"):
            self.assertNotIn(forbidden, HISTORY_MODEL_INPUT_FIELDS)

    def test_history_padding_and_length_mask(self) -> None:
        first = compact([0], 1, 1)
        second = compact([1], 1, 1)
        first["history"] = [[0.1] * HISTORY_DIM]
        second["history"] = [[0.2] * HISTORY_DIM, [0.3] * HISTORY_DIM]
        batch = collate_rows([first, second])
        self.assertEqual(batch["history_lengths"].tolist(), [1, 2])
        self.assertEqual(batch["history_mask"].tolist(), [[True, False], [True, True]])
        model = MaskedPointerActorCritic(16, history_encoder=True).eval()
        before = model.encode_history(batch["histories"], batch["history_lengths"])
        changed = batch["histories"].clone()
        changed[0, 1] = 1000.0
        after = model.encode_history(changed, batch["history_lengths"])
        self.assertTrue(torch.equal(before[0], after[0]))

    def test_stateless_default_remains_backward_compatible(self) -> None:
        torch.manual_seed(123)
        default = MaskedPointerActorCritic(16).eval()
        torch.manual_seed(123)
        explicit = MaskedPointerActorCritic(16, history_encoder=False).eval()
        self.assertEqual(default.state_dict().keys(), explicit.state_dict().keys())
        states = torch.randn(2, STATE_DIM)
        options = torch.randn(2, 3, ACTION_DIM)
        before = default(states, options)
        after = explicit(states, options)
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertTrue(torch.equal(before[1], after[1]))

    def test_two_arms_have_distinct_fingerprints_and_do_not_aggregate(self) -> None:
        common = dict(
            code_commit="code", input_sha256="input", split_seed=20260720,
            validation_fraction=0.2, epochs=60, batch_size=256, learning_rate=3e-4,
            hidden_dim=128, patience=10, value_loss_weight=0.25, gradient_clip_norm=1.0,
        )
        stateless = experiment_fingerprint(actual_fingerprint_payload(**common, architecture=ARCHITECTURE))
        history = experiment_fingerprint(actual_fingerprint_payload(
            **common, architecture=HISTORY_ARCHITECTURE, history_length=16
        ))
        self.assertNotEqual(stateless, history)
        weighted = experiment_fingerprint(actual_fingerprint_payload(
            **common,
            architecture=ARCHITECTURE,
            sampling={"recency_half_life_days": 2.0},
        ))
        self.assertNotEqual(stateless, weighted)
        rows = [
            {"seed": "17", "experiment_fingerprint": stateless, "config_matched": "True"},
            {"seed": "42", "experiment_fingerprint": history, "config_matched": "True"},
        ]
        self.assertEqual(achieved_seeds_for_fingerprint(rows, stateless), {17})
        reports = [
            {"experiment_id": "RL-BC-002-A", "provenance": {"experiment_fingerprint": stateless, "git_sha": "code", "input_sha256": "input"}},
            {"experiment_id": "RL-BC-002-B", "provenance": {"experiment_fingerprint": history, "git_sha": "code", "input_sha256": "input"}},
        ]
        with self.assertRaises(ValueError):
            aggregate_reports(reports, [17, 42])

    def test_history_checkpoint_reload_has_identical_output(self) -> None:
        torch.manual_seed(19)
        model = MaskedPointerActorCritic(16, history_encoder=True).eval()
        states = torch.randn(2, STATE_DIM)
        options = torch.randn(2, 3, ACTION_DIM)
        histories = torch.randn(2, 4, HISTORY_DIM)
        lengths = torch.tensor([2, 4])
        before = model(states, options, histories, lengths)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.pt"
            torch.save({"model": model.state_dict()}, path)
            restored = MaskedPointerActorCritic(16, history_encoder=True).eval()
            restored.load_state_dict(torch.load(path, weights_only=False)["model"])
        after = restored(states, options, histories, lengths)
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertTrue(torch.equal(before[1], after[1]))

    def test_history_decode_is_always_legal(self) -> None:
        rows = [compact([], 0, 3), compact([1], 1, 1), compact([0, 2], 2, 3)]
        for index, row in enumerate(rows):
            row["history"] = [[float(index)] * HISTORY_DIM]
        batch = collate_rows(rows)
        predictions = greedy_decode(MaskedPointerActorCritic(16, history_encoder=True), batch)
        for row, prediction in zip(rows, predictions):
            self.assertTrue(action_is_legal(prediction, len(row["options"]), row["min_count"], row["max_count"]))

    def test_resume_restores_best_history_and_early_stopping_state(self) -> None:
        state = new_training_state()
        state.update({"best_loss": 0.4, "best_epoch": 3, "stale": 2, "completed_epochs": 5, "optimizer_steps": 77, "best_checkpoint_path": "best.pt"})
        state["history"] = [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}, {"epoch": 4}, {"epoch": 5}]
        restored = restore_training_state({"training_state": state})
        self.assertEqual(restored, state)
        restored["history"].append({"epoch": 6})
        self.assertEqual(len(state["history"]), 5)

    def test_worse_resumed_epoch_does_not_replace_best(self) -> None:
        state = new_training_state()
        state.update({"best_loss": 0.4, "best_epoch": 2, "completed_epochs": 2, "best_checkpoint_path": "original_best.pt"})
        record = {"epoch": 3, "validation": {"loss": 0.6}}
        improved = update_training_state(state, record, full_epoch=True, optimizer_steps=10, best_path=Path("new_best.pt"))
        self.assertFalse(improved)
        self.assertEqual(state["best_checkpoint_path"], "original_best.pt")
        self.assertEqual(state["best_loss"], 0.4)
        self.assertEqual(state["stale"], 1)

    def test_smoke_batch_does_not_complete_epoch(self) -> None:
        state = new_training_state()
        record = {"epoch": 1, "full_epoch": False, "validation": {"loss": 1.0}}
        improved = update_training_state(state, record, full_epoch=False, optimizer_steps=1, best_path=Path("best.pt"))
        self.assertFalse(improved)
        self.assertEqual(state["completed_epochs"], 0)
        self.assertEqual(state["history"], [])
        self.assertEqual(len(state["partial_history"]), 1)
        self.assertEqual(state["optimizer_steps"], 1)

    def test_evaluation_input_sha_mismatch_fails_by_default(self) -> None:
        with self.assertRaises(ValueError):
            validate_input_sha({"input_sha256": "expected"}, "actual")
        validate_input_sha({"input_sha256": "expected"}, "actual", allow_mismatch=True)

    def test_dirty_worktree_blocks_formal_but_not_smoke(self) -> None:
        with patch("scripts.train_rl_policy.git_status_porcelain", return_value=" M results/file.json\n"):
            with self.assertRaises(RuntimeError):
                assert_formal_worktree_clean(0)
            self.assertIn("results/file.json", assert_formal_worktree_clean(1))

    @staticmethod
    def planned_config() -> dict:
        return {
            "input_sha256": "input-sha",
            "architecture": ARCHITECTURE,
            "split": {"seed": 20260720, "validation_fraction": 0.2},
            "training": {
                "formal_seeds": [17, 42, 20260720], "epochs": 30, "batch_size": 256,
                "learning_rate": 3e-4, "hidden_dim": 128, "early_stopping_patience": 5,
                "value_loss_weight": 0.25, "gradient_clip_norm": 1.0,
            },
        }

    def test_full_planned_config_match_and_batch_mismatch(self) -> None:
        actual = actual_fingerprint_payload(
            code_commit="code-a", input_sha256="input-sha", split_seed=20260720,
            validation_fraction=0.2, epochs=30, batch_size=256, learning_rate=3e-4,
            hidden_dim=128, patience=5, value_loss_weight=0.25, gradient_clip_norm=1.0,
        )
        self.assertEqual(current_config_matches(self.planned_config(), actual), (True, []))
        actual["training"]["batch_size"] = 512
        matched, mismatches = current_config_matches(self.planned_config(), actual)
        self.assertFalse(matched)
        self.assertEqual(mismatches, ["batch_size"])

    def test_cross_commit_or_config_seeds_are_not_merged(self) -> None:
        base = actual_fingerprint_payload(
            code_commit="code-a", input_sha256="input-sha", split_seed=20260720,
            validation_fraction=0.2, epochs=30, batch_size=256, learning_rate=3e-4,
            hidden_dim=128, patience=5, value_loss_weight=0.25, gradient_clip_norm=1.0,
        )
        wanted = experiment_fingerprint(base)
        other_commit = dict(base); other_commit["code_commit"] = "code-b"
        other_config = actual_fingerprint_payload(**{
            "code_commit": "code-a", "input_sha256": "input-sha", "split_seed": 20260720,
            "validation_fraction": 0.2, "epochs": 30, "batch_size": 512, "learning_rate": 3e-4,
            "hidden_dim": 128, "patience": 5, "value_loss_weight": 0.25, "gradient_clip_norm": 1.0,
        })
        rows = [
            {"seed": "17", "experiment_fingerprint": wanted, "config_matched": "True"},
            {"seed": "42", "experiment_fingerprint": experiment_fingerprint(other_commit), "config_matched": "True"},
            {"seed": "20260720", "experiment_fingerprint": experiment_fingerprint(other_config), "config_matched": "False"},
        ]
        self.assertEqual(achieved_seeds_for_fingerprint(rows, wanted), {17})

    def test_resume_checks_commit_input_and_fingerprint(self) -> None:
        checkpoint = {"git_sha": "code", "input_sha256": "input", "experiment_fingerprint": "fingerprint"}
        validate_resume_compatibility(checkpoint, code_commit="code", input_sha256="input", fingerprint="fingerprint")
        for key, kwargs in (
            ("git", {"code_commit": "other", "input_sha256": "input", "fingerprint": "fingerprint"}),
            ("input", {"code_commit": "code", "input_sha256": "other", "fingerprint": "fingerprint"}),
            ("fingerprint", {"code_commit": "code", "input_sha256": "input", "fingerprint": "other"}),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_resume_compatibility(checkpoint, **kwargs)


if __name__ == "__main__":
    unittest.main()
