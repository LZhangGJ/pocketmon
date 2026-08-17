from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


INTEGRATION = Path(__file__).resolve().parent
REFERENCE_TRAINING = INTEGRATION.parent / "reference" / "training"
for path in (INTEGRATION, REFERENCE_TRAINING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import Experiment7Error  # noqa: E402
from train_universal_bc_async_scaled import (  # noqa: E402
    initialize_scaled_model,
    scaled_model_config_kwargs,
    validate_source_contract,
)
from universal_deck_model import (  # noqa: E402
    UniversalDeckModelConfig,
    UniversalDeckTransformerPolicy,
)


class ScaledUniversalBCConfigTest(unittest.TestCase):
    def args(self, expected_card_vocab: int = 1268) -> argparse.Namespace:
        return argparse.Namespace(
            expected_card_vocab=expected_card_vocab,
            d_model=512,
            heads=8,
            layers=8,
            ff_dim=2048,
            dropout=0.05,
        )

    def test_source_card_vocab_produces_target_parameter_count(self) -> None:
        kwargs = scaled_model_config_kwargs(
            {"engineCatalog": {"cardVocab": 1268}}, self.args()
        )
        self.assertEqual(kwargs["card_vocab"], 1268)
        model = UniversalDeckTransformerPolicy(UniversalDeckModelConfig(**kwargs))
        self.assertEqual(model.parameter_count, 30_724_612)

    def test_expected_card_vocab_mismatch_fails_preflight(self) -> None:
        with self.assertRaisesRegex(Experiment7Error, "does not match expected"):
            scaled_model_config_kwargs(
                {"engineCatalog": {"cardVocab": 1269}}, self.args()
            )

    def test_strict_score_window_is_accepted_only_at_exact_contract(self) -> None:
        strict = {
            "kind": "experiment7_universal_bc_strict_score_window",
            "minGameScoreExclusive": 1000.0,
            "policySource": "winners",
        }
        validate_source_contract(strict)
        strict["minGameScoreExclusive"] = 900.0
        with self.assertRaisesRegex(Experiment7Error, ">1000"):
            validate_source_contract(strict)

    def test_initialize_scaled_model_requires_exact_config_and_state(self) -> None:
        model = mock.Mock()
        model.config.to_dict.return_value = {"card_vocab": 1268, "d_model": 512}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "epoch_000002.pt"
            checkpoint.write_bytes(b"frozen-checkpoint")
            payload = {
                "config": {"card_vocab": 1268, "d_model": 512},
                "state_dict": {"weight": torch.tensor([1.0])},
            }
            with mock.patch(
                "train_universal_bc_async_scaled.core.load_checkpoint",
                return_value=payload,
            ):
                receipt = initialize_scaled_model(
                    model, checkpoint, torch.device("cpu")
                )
        model.load_state_dict.assert_called_once_with(payload["state_dict"], strict=True)
        self.assertEqual(receipt["path"], str(checkpoint.resolve()))
        self.assertTrue(receipt["strict"])

    def test_initialize_scaled_model_rejects_config_mismatch(self) -> None:
        model = mock.Mock()
        model.config.to_dict.return_value = {"card_vocab": 1268, "d_model": 512}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "epoch_000002.pt"
            checkpoint.write_bytes(b"frozen-checkpoint")
            with mock.patch(
                "train_universal_bc_async_scaled.core.load_checkpoint",
                return_value={
                    "config": {"card_vocab": 1268, "d_model": 256},
                    "state_dict": {},
                },
            ):
                with self.assertRaisesRegex(Experiment7Error, "does not match"):
                    initialize_scaled_model(model, checkpoint, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
