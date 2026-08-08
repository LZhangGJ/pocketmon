from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "experiment7" / "integration"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, INTEGRATION / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(INTEGRATION))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


class Experiment7IntegrationTest(unittest.TestCase):
    def test_canonical_deck_is_permutation_invariant_and_multiplicity_sensitive(self) -> None:
        common = load("common")
        deck = list(range(1, 61))
        self.assertEqual(common.canonical_deck_sha256(deck), common.canonical_deck_sha256(list(reversed(deck))))
        changed = deck.copy()
        changed[-1] = changed[-2]
        self.assertNotEqual(common.canonical_deck_sha256(deck), common.canonical_deck_sha256(changed))

    def test_representative_deck_parser_requires_exactly_60_cards(self) -> None:
        selector = load("select_initial_decks")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "representative_decklists.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "archetype_rank",
                        "archetype_id",
                        "archetype_label",
                        "representative_exact_deck_id",
                        "card_id",
                        "count",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "archetype_rank": 1,
                        "archetype_id": "A01",
                        "archetype_label": "example",
                        "representative_exact_deck_id": "deck_x",
                        "card_id": 1,
                        "count": 60,
                    }
                )
            parsed = selector._load_representatives(path)
            self.assertEqual(len(parsed["A01"]["cards"]), 60)

    def test_wilson_gate_is_strictly_above_half(self) -> None:
        summary = load("summarize_challenger_results")
        low, high = summary.wilson(120.0, 200)
        self.assertGreater(low, 0.5)
        self.assertGreater(high, low)


if __name__ == "__main__":
    unittest.main()

class Experiment7ReferenceModelTest(unittest.TestCase):
    def test_reference_model_forward_contract(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        import sys
        training = ROOT / "experiment7" / "reference_impl" / "training"
        sys.path.insert(0, str(training))
        try:
            from deck_identity_model import DeckIdentityModelConfig, PTCGDeckIdentityTransformerPolicy
            config = DeckIdentityModelConfig(
                card_vocab=1600,
                opponent_classes=3,
                d_model=32,
                n_heads=4,
                n_layers=1,
                ff_dim=64,
            )
            model = PTCGDeckIdentityTransformerPolicy(config)
            outputs = model(
                torch.zeros((2, 320)),
                torch.zeros((2, 8, 320)),
                torch.zeros((2, 8, 176)),
                torch.zeros((2, 8), dtype=torch.bool),
                torch.ones((2, 60), dtype=torch.long),
                torch.zeros((2, 3, 10), dtype=torch.long),
                torch.zeros((2, 3, 12)),
                torch.ones((2, 3), dtype=torch.bool),
                torch.zeros((2, 4, 176)),
                torch.ones((2, 4), dtype=torch.bool),
            )
            self.assertEqual(outputs[0].shape, (2, 4))
            self.assertEqual(outputs[1].shape, (2, 41))
            self.assertEqual(outputs[2].shape, (2,))
            self.assertEqual(outputs[3].shape, (2, 3))
        finally:
            sys.path.pop(0)
