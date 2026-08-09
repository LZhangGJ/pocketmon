from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "experiment7" / "integration"
REFERENCE = ROOT / "experiment7" / "reference"
for path in (INTEGRATION, REFERENCE / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import canonical_deck_sha256, directory_sha256


class Experiment7IntegrationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
