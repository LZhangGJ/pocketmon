from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath

import torch

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "experiment7" / "integration"
REFERENCE = ROOT / "experiment7" / "reference"
for path in (INTEGRATION, REFERENCE / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import canonical_deck_sha256, directory_sha256, write_csv
from arena import make_schedule
from build_from_pocketmon_replays import selected_rows
from multi_gpu_scheduler import make_specialist_plan
from prepare_universal_training_data import reuse_engine_catalog
from export_and_package import select_universal
from universal_deck_model import (
    UniversalDeckModelConfig,
    UniversalDeckTransformerPolicy,
    universal_bc_loss,
)
from universal_deck_portable import PortableUniversalDeckTransformerPolicy, _stable_argmax
from verify_universal_portable import stable_order


class Experiment7IntegrationTests(unittest.TestCase):
    def test_replay_score_filter_is_strictly_greater_than_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory) / "catalog.csv"
            write_csv(
                catalog,
                [
                    {
                        "episode_id": episode,
                        "create_time": f"2026-08-08T00:00:0{index}Z",
                        "raw_path": f"/{episode}.json",
                        "is_clean": "1",
                        "module_version": "1.32.6",
                        "min_score": score,
                        "policy_weight0": "1",
                        "policy_weight1": "0",
                    }
                    for index, (episode, score) in enumerate(
                        ((1, "899.9"), (2, "900"), (3, "900.1"), (4, "1200"))
                    )
                ],
            )
            rows = selected_rows(
                catalog,
                "broad",
                None,
                None,
                False,
                0,
                min_game_score_exclusive=900.0,
            )
            self.assertEqual([int(row["episode_id"]) for row in rows], [3, 4])

    def test_universal_prepare_reuses_verified_engine_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            output = root / "output.json"
            source.write_text(
                json.dumps(
                    {
                        "cards": [{"cardId": 1}, {"cardId": 7}],
                        "attacks": [{"attackId": 3}],
                    }
                ),
                encoding="utf-8",
            )
            receipt = reuse_engine_catalog(source, output)
            self.assertEqual(receipt["cardVocab"], 8)
            self.assertEqual(receipt["cards"], 2)
            self.assertEqual(receipt["attacks"], 1)
            self.assertEqual(receipt["sourceSha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertTrue(output.is_file())

    def test_specialist_plan_assigns_one_exact_source_per_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / "inventory.json"
            inventory.write_text(
                json.dumps(
                    {
                        "gpus": [
                            {
                                "host": "doraemon15",
                                "gpuIndex": index,
                                "name": "RTX 8000",
                                "totalMiB": 48601,
                                "usedMiB": 1,
                                "freeMiB": 48600,
                                "utilizationPercent": 0,
                                "eligible": True,
                            }
                            for index in range(2)
                        ]
                    }
                ),
                encoding="utf-8",
            )
            plan = make_specialist_plan(
                inventory,
                root / "plan.json",
                PurePosixPath("/homes/lzhang/worktrees/fixed"),
                "a" * 40,
                "/env/bin/python",
                PurePosixPath("/runs/prepared/training_sources.json"),
                PurePosixPath("/runs/specialists"),
                PurePosixPath("/runs/training/best_model.pt"),
                ["01_a01_hash", "02_a02_hash"],
                20260809,
            )
            self.assertEqual([job["gpuIndex"] for job in plan["jobs"]], [0, 1])
            self.assertEqual(
                [job["command"][2] for job in plan["jobs"]],
                ["specialize", "specialize"],
            )
            self.assertIn("01_a01_hash", plan["jobs"][0]["command"])
            self.assertIn("02_a02_hash", plan["jobs"][1]["command"])

    def test_reference_manifest_is_complete(self) -> None:
        manifest = REFERENCE / "PACKAGE_MANIFEST.csv"
        self.assertTrue(manifest.is_file())
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 38)
        for row in rows:
            normalized = {str(key).strip().lower(): value for key, value in row.items()}
            path = REFERENCE / normalized["path"]
            self.assertTrue(path.is_file(), normalized["path"])
            self.assertEqual(path.stat().st_size, int(normalized["bytes"]))
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest().lower(),
                normalized["sha256"].lower(),
            )

    def test_exact_deck_hash_is_order_invariant_and_count_sensitive(self) -> None:
        first = list(range(1, 61))
        second = list(reversed(first))
        third = first.copy()
        third[-1] = third[-2]
        self.assertEqual(canonical_deck_sha256(first), canonical_deck_sha256(second))
        self.assertNotEqual(canonical_deck_sha256(first), canonical_deck_sha256(third))

    def test_default_config_has_remote_execution_context(self) -> None:
        payload = json.loads(
            (ROOT / "experiment7" / "configs" / "multideck_default.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["repository"], "LZhangGJ/pocketmon")
        self.assertEqual(
            payload["sourceBranch"], "agent/experiment7-training-ready-20260809"
        )
        self.assertEqual(
            payload["workBranch"], "codex/experiment7-multideck-run-20260809"
        )
        self.assertGreaterEqual(len(payload["linux"]["servers"]), 6)
        self.assertEqual(payload["model"]["historyLength"], 8)
        self.assertEqual(payload["deckSelection"]["desired"], 6)

    def test_runtime_directory_hash_ignores_python_caches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "main.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            original = directory_sha256(root)
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "main.cpython-311.pyc").write_bytes(b"generated-cache")
            (root / "orphan.pyc").write_bytes(b"generated-cache")
            self.assertEqual(directory_sha256(root), original)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(directory_sha256(root), original)

    def test_integration_csv_has_no_byte_order_mark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "schedule.csv"
            write_csv(output, [{"game_id": "smoke-1"}], ["game_id"])
            self.assertFalse(output.read_bytes().startswith(b"\xef\xbb\xbf"))
            with output.open(encoding="utf-8", newline="") as handle:
                self.assertEqual(next(csv.DictReader(handle))["game_id"], "smoke-1")

    def test_arena_schedule_columns_match_existing_league_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            (target / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
            packages = root / "packages.json"
            packages.write_text(
                json.dumps(
                    {
                        "packages": [
                            {"name": "challenger", "agentDir": str(root / "agent")}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "smoke"
            make_schedule(packages, target, output, 4, 100, "smoke", None)
            with (output / "schedule.csv").open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(
                    reader.fieldnames,
                    ["learner", "opponent", "seed", "learner_seat"],
                )
                self.assertEqual(len(list(reader)), 4)

    def test_reference_model_deck_multiset_invariance(self) -> None:
        from deck_identity_model import (
            DeckIdentityModelConfig,
            PTCGDeckIdentityTransformerPolicy,
        )

        torch.manual_seed(20260808)
        config = DeckIdentityModelConfig(
            d_model=32,
            n_heads=4,
            n_layers=1,
            ff_dim=64,
            dropout=0.0,
            opponent_classes=3,
        )
        model = PTCGDeckIdentityTransformerPolicy(config).eval()
        batch = 1
        history = config.history_length
        entities = 2
        actions = 3
        state = torch.randn(batch, config.state_dim)
        history_state = torch.randn(batch, history, config.state_dim)
        history_action = torch.randn(batch, history, config.option_dim)
        history_mask = torch.ones(batch, history, dtype=torch.bool)
        deck = torch.tensor([list(range(1, 61))], dtype=torch.long)
        entity_cat = torch.zeros(batch, entities, 10, dtype=torch.long)
        entity_cat[:, :, 0] = torch.tensor([[1, 2]])
        entity_cat[:, 1, 2] = 1
        entity_num = torch.randn(batch, entities, config.entity_num_dim)
        entity_mask = torch.ones(batch, entities, dtype=torch.bool)
        options = torch.randn(batch, actions, config.option_dim)
        option_mask = torch.ones(batch, actions, dtype=torch.bool)

        args = (
            state,
            history_state,
            history_action,
            history_mask,
            deck,
            entity_cat,
            entity_num,
            entity_mask,
            options,
            option_mask,
        )
        with torch.inference_mode():
            original = model(*args)[0]
            permuted = list(args)
            permuted[4] = deck[:, torch.randperm(60)]
            reordered = model(*permuted)[0]
            changed = list(args)
            changed_deck = deck.clone()
            changed_deck[0, -1] = changed_deck[0, -2]
            changed[4] = changed_deck
            different = model(*changed)[0]
        self.assertTrue(torch.allclose(original, reordered, atol=1e-6, rtol=0.0))
        self.assertFalse(torch.allclose(original, different, atol=1e-6, rtol=0.0))

    def _universal_batch(self):
        config = UniversalDeckModelConfig(
            state_dim=12,
            option_dim=8,
            entity_num_dim=4,
            card_vocab=128,
            d_model=32,
            n_heads=4,
            n_layers=1,
            ff_dim=64,
            max_actions=6,
            history_length=2,
            deck_size=6,
            deck_latents=8,
            dropout=0.0,
        )
        model = UniversalDeckTransformerPolicy(config).eval()
        batch_size, entities, options = 2, 3, 4
        inputs = (
            torch.randn(batch_size, config.state_dim),
            torch.randn(batch_size, config.history_length, config.state_dim),
            torch.randn(batch_size, config.history_length, config.option_dim),
            torch.ones(batch_size, config.history_length, dtype=torch.bool),
            torch.tensor([[1, 2, 2, 3, 4, 5], [7, 7, 8, 9, 10, 11]]),
            torch.zeros(batch_size, entities, 10, dtype=torch.long),
            torch.randn(batch_size, entities, config.entity_num_dim),
            torch.ones(batch_size, entities, dtype=torch.bool),
            torch.randn(batch_size, options, config.option_dim),
            torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool),
        )
        batch = {
            "original_labels": torch.tensor(
                [[0, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.float32
            ),
            "min_count": torch.tensor([1, 0]),
            "max_count": torch.tensor([2, 1]),
            "winner": torch.tensor([1.0, 0.0]),
        }
        return model, inputs, batch

    def test_universal_deck8_is_multiset_invariant_and_count_sensitive(self) -> None:
        torch.manual_seed(20260810)
        model, inputs, _ = self._universal_batch()
        with torch.inference_mode():
            original = model(*inputs)
            permuted_inputs = list(inputs)
            permuted_inputs[4] = inputs[4][:, torch.tensor([5, 2, 0, 4, 1, 3])]
            permuted = model(*permuted_inputs)
            changed_inputs = list(inputs)
            changed_deck = inputs[4].clone()
            changed_deck[0, -1] = changed_deck[0, -2]
            changed_inputs[4] = changed_deck
            changed = model(*changed_inputs)
        self.assertEqual(tuple(original.deck_hidden.shape), (2, 8, 32))
        self.assertTrue(
            torch.allclose(original.option_hidden, permuted.option_hidden, atol=1e-6, rtol=0.0)
        )
        self.assertFalse(
            torch.allclose(original.option_hidden, changed.option_hidden, atol=1e-6, rtol=0.0)
        )

    def test_universal_joint_decoder_enforces_stop_and_selection_legality(self) -> None:
        torch.manual_seed(20260810)
        model, inputs, batch = self._universal_batch()
        encoding = model(*inputs)
        selected = torch.zeros_like(inputs[-1], dtype=torch.bool)
        logits = model.decoder_logits(
            encoding, selected, batch["min_count"], batch["max_count"]
        )
        self.assertLess(float(logits[0, -1]), -9999.0)  # STOP before minCount
        self.assertGreater(float(logits[1, -1]), -9999.0)
        selected[0, 0] = True
        selected[0, 1] = True
        at_max = model.decoder_logits(
            encoding, selected, batch["min_count"], batch["max_count"]
        )
        self.assertTrue(bool(torch.all(at_max[0, :-1] <= -9999.0)))
        actions = model.greedy_actions(
            encoding, batch["min_count"], batch["max_count"]
        )
        self.assertTrue(all(1 <= len(actions[0]) <= 2 for _ in [0]))
        self.assertTrue(all(0 <= len(actions[1]) <= 1 for _ in [0]))
        self.assertEqual(len(actions[0]), len(set(actions[0])))

    def test_universal_portable_matches_deck8_autoregressive_actions(self) -> None:
        import numpy as np

        torch.manual_seed(20260810)
        model, inputs, batch = self._universal_batch()
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            portable_path = Path(directory) / "universal.npz"
            arrays = {
                name: value.detach().numpy().astype(np.float32, copy=False)
                for name, value in model.state_dict().items()
            }
            arrays["config_json"] = np.asarray(
                [json.dumps(model.config.to_dict(), separators=(",", ":"), sort_keys=True)]
            )
            np.savez_compressed(portable_path, **arrays)
            portable = PortableUniversalDeckTransformerPolicy(portable_path)
            with torch.inference_mode():
                torch_encoding = model(*inputs)
                torch_action = model.greedy_actions(
                    torch_encoding, batch["min_count"], batch["max_count"]
                )[0]
            portable_encoding = portable.encode(
                inputs[0][0].numpy(),
                inputs[1][0].numpy(),
                inputs[2][0].numpy(),
                inputs[3][0].numpy(),
                inputs[4][0].numpy(),
                inputs[5][0].numpy(),
                inputs[6][0].numpy(),
                inputs[7][0].numpy(),
                inputs[8][0].numpy(),
                inputs[9][0].numpy(),
            )
            portable_action = portable.greedy_actions(
                portable_encoding,
                int(batch["min_count"][0]),
                int(batch["max_count"][0]),
            )
            self.assertEqual(portable_action, torch_action)
            self.assertTrue(
                np.allclose(
                    portable_encoding["option_hidden"],
                    torch_encoding.option_hidden[0].numpy(),
                    atol=2e-3,
                    rtol=0.0,
                )
            )

    def test_universal_portable_near_tie_prefers_lower_index(self) -> None:
        import numpy as np

        logits = np.asarray([-2.0, -1.00010, -3.0, -1.0], dtype=np.float32)
        self.assertEqual(_stable_argmax(logits), 1)

    def test_universal_stable_order_groups_numerical_near_ties(self) -> None:
        import numpy as np

        logits = np.asarray([2.0, -1.0003, -3.0, -1.0], dtype=np.float32)
        valid = np.ones(4, dtype=bool)
        self.assertEqual(stable_order(logits, valid).tolist(), [0, 1, 3, 2])

    def test_select_universal_shortlists_best_validation_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, semantic, index in ((11, 0.75, 0.72), (12, 0.76, 0.71)):
                run = root / f"seed-{seed}"
                run.mkdir()
                checkpoint = run / "best_model.pt"
                checkpoint.write_bytes(f"checkpoint-{seed}".encode())
                (run / "training_report.json").write_text(
                    json.dumps(
                        {
                            "stage": "universal_bc",
                            "seed": seed,
                            "best": {"epoch": 4, "path": str(checkpoint)},
                            "epochs": [
                                {
                                    "epoch": 4,
                                    "validation": {
                                        "exactSemantic": semantic,
                                        "exactIndex": index,
                                        "countAccuracy": 0.99,
                                        "illegalPredictionCount": 0,
                                    },
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            output = root / "selection.json"
            payload = select_universal(root, output, 2)
            self.assertEqual(payload["selected"]["seed"], 12)
            self.assertEqual([row["seed"] for row in payload["shortlist"]], [12, 11])
            self.assertFalse(payload["holdoutUsed"])

    def test_universal_value_loss_uses_loser_when_policy_weight_is_zero(self) -> None:
        torch.manual_seed(20260810)
        model, inputs, batch = self._universal_batch()
        encoding = model(*inputs)
        loss, parts = universal_bc_loss(
            model,
            encoding,
            batch,
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, 1.0]),
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(int(parts["policyExamples"]), 1)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        loser_only = torch.nn.functional.binary_cross_entropy_with_logits(
            encoding.value_logits[1], batch["winner"][1]
        )
        self.assertGreater(float(loser_only), 0.0)


if __name__ == "__main__":
    unittest.main()
