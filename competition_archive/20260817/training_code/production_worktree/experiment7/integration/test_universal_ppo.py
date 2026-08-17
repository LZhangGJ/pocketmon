from __future__ import annotations

import os
import sys
import tempfile
import unittest
import gzip
import json
from pathlib import Path

import numpy as np
import torch


INTEGRATION = Path(__file__).resolve().parent
REFERENCE = INTEGRATION.parent / "reference"
for path in (INTEGRATION, REFERENCE / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from universal_deck_model import (  # noqa: E402
    UniversalDeckModelConfig,
    UniversalDeckTransformerPolicy,
    stable_torch_argmax,
)
from universal_ppo import (  # noqa: E402
    collate_rows,
    compute_gae,
    evaluate_actions,
    normalize_advantages,
    ppo_loss,
    sample_action,
)
from train_universal_ppo import (  # noqa: E402
    balanced_player_order,
    compact_parent_metadata,
    load_rollouts,
    require_clean_repository,
)
from collect_universal_ppo_rollouts import canonical_archetype, finish_trajectory  # noqa: E402
from ppo_tactical_shaping import tactical_adjustment  # noqa: E402
from common import Experiment7Error  # noqa: E402
from export_and_package import add_runtime_diagnostics  # noqa: E402
from universal_ppo import ROLLOUT_FORMAT  # noqa: E402


def row(action: list[int] | None = None) -> dict:
    config = UniversalDeckModelConfig(d_model=32, n_heads=4, n_layers=1, ff_dim=64)
    return {
        "state": np.zeros(config.state_dim, dtype=np.float32).tolist(),
        "history_state": np.zeros((config.history_length, config.state_dim), dtype=np.float32).tolist(),
        "history_action": np.zeros((config.history_length, config.option_dim), dtype=np.float32).tolist(),
        "history_mask": np.zeros(config.history_length, dtype=np.uint8).tolist(),
        "own_deck_cards": ([7, 104, 112] * 20),
        "entity_cat": np.zeros((4, 10), dtype=np.int64).tolist(),
        "entity_num": np.zeros((4, config.entity_num_dim), dtype=np.float32).tolist(),
        "entity_mask": [1, 1, 0, 0],
        "options": np.zeros((3, config.option_dim), dtype=np.float32).tolist(),
        "min_count": 1,
        "max_count": 2,
        "action": list(action or []),
    }


class UniversalPpoTest(unittest.TestCase):
    def test_stable_torch_argmax_matches_portable_near_tie_rule(self) -> None:
        logits = torch.tensor(
            [
                [1.0, 1.0004, 0.0],
                [1.0, 1.0006, 0.0],
            ],
            dtype=torch.float32,
        )
        self.assertEqual(stable_torch_argmax(logits).tolist(), [0, 1])

    def setUp(self) -> None:
        torch.manual_seed(17)
        config = UniversalDeckModelConfig(d_model=32, n_heads=4, n_layers=1, ff_dim=64)
        self.model = UniversalDeckTransformerPolicy(config)
        self.device = torch.device("cpu")

    def test_sample_and_recompute_log_probability(self) -> None:
        action, log_probability, value, entropy = sample_action(
            self.model.eval(), row(), self.device
        )
        prepared = row(action)
        recomputed, recomputed_entropy, recomputed_value = evaluate_actions(
            self.model, collate_rows([prepared], self.device)
        )
        self.assertAlmostEqual(float(recomputed[0]), log_probability, places=5)
        self.assertAlmostEqual(float(recomputed_value[0]), value, places=5)
        self.assertAlmostEqual(float(recomputed_entropy[0]), entropy, places=5)

    def test_ppo_loss_backpropagates(self) -> None:
        rows = []
        for index in range(2):
            action, log_probability, value, entropy = sample_action(
                self.model.eval(), row(), self.device
            )
            item = row(action)
            item.update(
                {
                    "episode_id": f"episode-{index}",
                    "player": 0,
                    "action_step": 1,
                    "behavior_log_probability": log_probability,
                    "teacher_log_probability": log_probability,
                    "behavior_value": value,
                    "behavior_entropy": entropy,
                    "reward": 1.0 if index == 0 else -1.0,
                }
            )
            rows.append(item)
        prepared = normalize_advantages(compute_gae(rows))
        self.model.train()
        loss, metrics = ppo_loss(self.model, collate_rows(prepared, self.device))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))
        self.assertAlmostEqual(metrics["ratioMean"], 1.0, places=5)

    def test_gae_accepts_intermediate_tactical_rewards(self) -> None:
        rows = [
            {
                "episode_id": "shaped",
                "player": 0,
                "action_step": 1,
                "behavior_value": 0.0,
                "reward": -0.3,
            },
            {
                "episode_id": "shaped",
                "player": 0,
                "action_step": 2,
                "behavior_value": 0.0,
                "reward": 1.0,
            },
        ]
        prepared = compute_gae(rows, gamma=1.0, gae_lambda=1.0)
        self.assertAlmostEqual(prepared[0]["return"], 0.7, places=6)
        self.assertAlmostEqual(prepared[1]["return"], 1.0, places=6)

    def test_long_game_weight_is_attached_per_player_trajectory(self) -> None:
        rows = [
            {"player": 0, "outcome": 0.0, "tactical_reward": 0.0},
            {"player": 0, "outcome": 0.0, "tactical_reward": 0.0},
            {"player": 1, "outcome": 0.0, "tactical_reward": 0.0},
        ]
        finish_trajectory(
            rows,
            0,
            long_game_min_player_decisions=2,
            long_game_weight=1.4,
        )
        self.assertEqual([row["sample_weight"] for row in rows], [1.4, 1.4, 1.0])
        self.assertEqual([row["long_game_episode"] for row in rows], [True, True, False])

    def test_a08_penalizes_terminal_action_before_evolution(self) -> None:
        observation = {
            "current": {"yourIndex": 0, "players": [{"active": [], "bench": []}, {}]},
            "select": {},
        }
        options = [{"type": 9}, {"type": 13}, {"type": 14}]
        features = type("Features", (), {"resolve_option_cards": staticmethod(lambda *_: (None, None))})
        adjustment = tactical_adjustment(
            "a08", observation, options, [1], features=features, cards={}
        )
        self.assertEqual(adjustment.events, ("a08_terminal_before_evolve",))
        self.assertAlmostEqual(adjustment.reward, -0.35)
        self.assertEqual(adjustment.preferred_action, (0,))

    def test_a02_penalty_is_conditional_not_every_playable_card(self) -> None:
        observation = {
            "current": {
                "yourIndex": 0,
                "energyAttached": True,
                "players": [
                    {"active": [{"id": 1, "hp": 320}], "bench": [{"id": 2, "hp": 100}]},
                    {"active": [{"id": 5, "hp": 220}], "bench": []},
                ],
            },
            "select": {},
        }
        options = [
            {"type": 9, "card": {"id": 3}},
            {"type": 13},
        ]
        features = type(
            "Features",
            (),
            {"resolve_option_cards": staticmethod(lambda _obs, option: (option.get("card"), None))},
        )
        cards = {
            1: {"name": "Marnie's Grimmsnarl ex"},
            2: {"name": "Marnie's Morgrem"},
            3: {"name": "Marnie's Grimmsnarl ex"},
            5: {"name": "Mega Kangaskhan ex"},
        }
        adjustment = tactical_adjustment(
            "a02", observation, options, [1], features=features, cards=cards
        )
        self.assertIn("a02_terminal_before_second_grimmsnarl", adjustment.events)
        self.assertAlmostEqual(adjustment.reward, -0.30)
        self.assertEqual(adjustment.preferred_action, (0,))

    def test_a02_munkidori_ability_is_not_bench_overfill(self) -> None:
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {"active": [{"id": 1}], "bench": [{"id": 2}, {"id": 2}, {"id": 2}]},
                    {"active": [], "bench": []},
                ],
            },
            "select": {},
        }
        options = [{"type": 10, "card": {"id": 2}}, {"type": 14}]
        features = type(
            "Features",
            (),
            {"resolve_option_cards": staticmethod(lambda _obs, option: (option.get("card"), None))},
        )
        cards = {1: {"name": "Marnie's Grimmsnarl ex"}, 2: {"name": "Munkidori"}}
        adjustment = tactical_adjustment(
            "a02", observation, options, [0], features=features, cards=cards
        )
        self.assertNotIn("a02_overfilled_munkidori_bench", adjustment.events)

    def test_tactical_preference_ranking_contributes_to_loss(self) -> None:
        prepared = row([0])
        with torch.inference_mode():
            log_probability, _, value = evaluate_actions(
                self.model.eval(), collate_rows([prepared], self.device)
            )
        prepared.update(
            behavior_log_probability=float(log_probability[0]),
            teacher_log_probability=float(log_probability[0]),
            behavior_value=float(value[0]),
            advantage=1.0,
            player=0,
            tactical_preferred_action=[1],
        )
        prepared["return"] = float(value[0]) + 1.0
        loss, metrics = ppo_loss(
            self.model,
            collate_rows([prepared], self.device),
            tactical_preference_coefficient=0.1,
        )
        loss.backward()
        self.assertEqual(metrics["tacticalPreferenceRows"], 1.0)
        self.assertGreater(metrics["tacticalPreferenceLoss"], 0.0)

    def test_ppo_loss_can_upweight_second_seat(self) -> None:
        action, log_probability, value, entropy = sample_action(
            self.model.eval(), row(), self.device
        )
        rows = []
        for player, advantage in ((0, 1.0), (1, -1.0)):
            item = row(action)
            item.update(
                {
                    "episode_id": f"seat-{player}",
                    "player": player,
                    "action_step": 1,
                    "behavior_log_probability": log_probability,
                    "teacher_log_probability": log_probability,
                    "behavior_value": value,
                    "behavior_entropy": entropy,
                    "advantage": advantage,
                    "return": value,
                }
            )
            rows.append(item)
        _, balanced = ppo_loss(self.model, collate_rows(rows, self.device))
        _, weighted = ppo_loss(
            self.model, collate_rows(rows, self.device), seat1_weight=2.0
        )
        self.assertAlmostEqual(balanced["policyLoss"], 0.0, places=5)
        self.assertAlmostEqual(weighted["policyLoss"], 1.0 / 3.0, places=5)
        self.assertEqual(weighted["seat1Weight"], 2.0)

    def test_player_specific_advantage_normalization_and_balancing(self) -> None:
        rows = [
            {"player": 0, "advantage": 10.0},
            {"player": 0, "advantage": 12.0},
            {"player": 1, "advantage": -100.0},
            {"player": 1, "advantage": -80.0},
            {"player": 1, "advantage": -60.0},
        ]
        normalized = normalize_advantages(rows, by_player=True)
        for player in (0, 1):
            values = [row["advantage"] for row in normalized if row["player"] == player]
            self.assertAlmostEqual(float(np.mean(values)), 0.0, places=7)
        order = balanced_player_order(rows, np.random.default_rng(9))
        players = [rows[int(index)]["player"] for index in order]
        self.assertEqual(players[::2], [0, 0, 0])
        self.assertEqual(players[1::2], [1, 1, 1])

    def test_repository_validation_does_not_depend_on_process_cwd(self) -> None:
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.chdir(temporary)
                repository, commit = require_clean_repository(INTEGRATION / "train_universal_ppo.py")
        finally:
            os.chdir(previous)
        self.assertTrue((repository / ".git").exists() or (repository / ".git").is_file())
        self.assertEqual(len(commit), 40)

    def test_packaged_diagnostics_precede_kaggle_agent_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            main_py = Path(temporary) / "main.py"
            main_py.write_text(
                "def helper():\n    return None\n\ndef agent(obs):\n    return []\n",
                encoding="utf-8",
            )
            add_runtime_diagnostics(main_py)
            source = main_py.read_text(encoding="utf-8")
            self.assertLess(source.index("def diagnostics"), source.index("def agent"))

    def test_parent_metadata_does_not_embed_ancestor_chain(self) -> None:
        nested = {"metadata": {"generation": 490, "parentMetadata": {}}}
        cursor = nested["metadata"]["parentMetadata"]
        for generation in range(489, 0, -1):
            cursor["generation"] = generation
            cursor["parentMetadata"] = {}
            cursor = cursor["parentMetadata"]
        compact = compact_parent_metadata(nested)
        self.assertEqual(compact, {"generation": 490})
        with tempfile.TemporaryDirectory() as temporary:
            torch.save(compact, Path(temporary) / "ppo-compact-parent-test.pt")

    def test_canonical_archetype_collapses_b08_aliases(self) -> None:
        self.assertEqual(canonical_archetype({"archetype": "alakazam"}), "A03")
        self.assertEqual(canonical_archetype({"archetype": "dragapult"}), "A06")
        self.assertEqual(canonical_archetype({"archetype": "A08"}), "A08")
        self.assertEqual(
            canonical_archetype({"archetype": "A03", "canonical_archetype": "A07"}),
            "A07",
        )

    def test_async_rollouts_accept_bounded_staleness(self) -> None:
        item = row([0])
        item.update(
            {
                "rollout_format": ROLLOUT_FORMAT,
                "episode_id": "async-episode",
                "player": 0,
                "action_step": 1,
                "behavior_checkpoint_sha256": "old-sha",
                "behavior_generation": 8,
                "teacher_checkpoint_sha256": "teacher-sha",
                "behavior_log_probability": 0.0,
                "teacher_log_probability": 0.0,
                "behavior_value": 0.0,
                "behavior_entropy": 0.0,
                "reward": 1.0,
                "outcome": 1.0,
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rollout.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(item) + "\n")
            rows, _ = load_rollouts(
                [path],
                "current-sha",
                "teacher-sha",
                allowed_behavior_generations={"old-sha": 8},
                current_generation=10,
                max_behavior_lag=2,
            )
            self.assertEqual(len(rows), 1)
            with self.assertRaises(Experiment7Error):
                load_rollouts(
                    [path],
                    "current-sha",
                    "teacher-sha",
                    allowed_behavior_generations={"old-sha": 8},
                    current_generation=11,
                    max_behavior_lag=2,
                )


if __name__ == "__main__":
    unittest.main()
