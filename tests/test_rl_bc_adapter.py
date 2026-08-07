from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rl.agent_adapter import HISTORY_ARCHITECTURE, STATELESS_ARCHITECTURE, RLBCPolicyAdapter
from rl.bc import action_is_legal
from rl.model import MaskedPointerActorCritic
from scripts.run_local_match import agent_diagnostics


def observation(min_count: int = 0, max_count: int = 2, turn: int = 1) -> dict:
    return {
        "current": {"players": [{}, {}], "yourIndex": 0, "turn": turn},
        "select": {
            "type": 1,
            "minCount": min_count,
            "maxCount": max_count,
            "option": [{"type": 1}, {"type": 2}, {"type": 3}],
        },
    }


def save_checkpoint(path: Path, history: bool = False) -> None:
    torch.manual_seed(7)
    model = MaskedPointerActorCritic(16, history_encoder=history)
    torch.save({
        "model": model.state_dict(),
        "hidden_dim": 16,
        "config": {
            "architecture": HISTORY_ARCHITECTURE if history else STATELESS_ARCHITECTURE,
            "history_length": 2 if history else 0,
        },
    }, path)


class RLBCAdapterTests(unittest.TestCase):
    def test_model_decode_is_legal_and_uses_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            save_checkpoint(checkpoint)
            adapter = RLBCPolicyAdapter(checkpoint, fallback=lambda _: [])
            obs = observation(min_count=1, max_count=2)
            action = adapter.act(obs)
        self.assertTrue(action_is_legal(action, 3, 1, 2))
        self.assertEqual(adapter.diagnostics()["model_actions"], 1)
        self.assertEqual(adapter.diagnostics()["fallback_actions"], 0)

    def test_missing_checkpoint_uses_rule_fallback(self) -> None:
        adapter = RLBCPolicyAdapter(Path("missing.pt"), fallback=lambda _: [])
        action = adapter.act(observation(min_count=0, max_count=2))
        self.assertEqual(action, [])
        diagnostics = adapter.diagnostics()
        self.assertEqual(diagnostics["load_errors"], 1)
        self.assertEqual(diagnostics["fallback_actions"], 1)

    def test_illegal_fallback_is_replaced_by_emergency_legal_action(self) -> None:
        adapter = RLBCPolicyAdapter(Path("missing.pt"), fallback=lambda _: [])
        action = adapter.act(observation(min_count=2, max_count=2))
        self.assertEqual(action, [0, 1])
        diagnostics = adapter.diagnostics()
        self.assertEqual(diagnostics["illegal_fallback_actions"], 1)
        self.assertEqual(diagnostics["emergency_legal_actions"], 1)

    def test_inference_exception_uses_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            save_checkpoint(checkpoint)
            adapter = RLBCPolicyAdapter(checkpoint, fallback=lambda _: [1])
            with patch.object(adapter, "_model_action", side_effect=RuntimeError("boom")):
                action = adapter.act(observation(min_count=1, max_count=1))
        self.assertEqual(action, [1])
        self.assertEqual(adapter.diagnostics()["inference_errors"], 1)

    def test_history_is_appended_only_after_each_action_and_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "history.pt"
            save_checkpoint(checkpoint, history=True)
            adapter = RLBCPolicyAdapter(checkpoint, fallback=lambda _: [])
            self.assertEqual(adapter.diagnostics()["history_tokens"], 0)
            for turn in (1, 2, 3):
                action = adapter.act(observation(min_count=0, max_count=2, turn=turn))
                self.assertTrue(action_is_legal(action, 3, 0, 2))
        diagnostics = adapter.diagnostics()
        self.assertTrue(diagnostics["history_enabled"])
        self.assertEqual(diagnostics["history_tokens"], 2)

    def test_deck_request_resets_history_and_delegates(self) -> None:
        adapter = RLBCPolicyAdapter(Path("missing.pt"), fallback=lambda _: [11, 12])
        adapter._history = [[1.0], [2.0]]
        self.assertEqual(adapter.act({"current": {}, "select": None}), [11, 12])
        self.assertEqual(adapter.diagnostics()["history_tokens"], 0)

    def test_local_runner_collects_adapter_diagnostics(self) -> None:
        class WithDiagnostics:
            @staticmethod
            def diagnostics():
                return {"fallback_actions": 2}

        self.assertEqual(agent_diagnostics([WithDiagnostics(), object()]), [{"fallback_actions": 2}, {}])


if __name__ == "__main__":
    unittest.main()
