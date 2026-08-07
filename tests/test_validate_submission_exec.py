from __future__ import annotations

import unittest

from scripts.validate_submission_exec import _validate_model_diagnostics


def valid_diagnostics() -> dict[str, object]:
    return {
        "checkpoint_exists": True,
        "model_actions": 1,
        "load_errors": 0,
        "inference_errors": 0,
        "illegal_model_actions": 0,
        "fallback_actions": 0,
        "illegal_fallback_actions": 0,
        "emergency_legal_actions": 0,
        "q_load_errors": 0,
    }


class ValidateSubmissionExecTests(unittest.TestCase):
    def test_accepts_loaded_checkpoint_and_clean_model_decision(self) -> None:
        diagnostics = valid_diagnostics()
        self.assertIs(_validate_model_diagnostics(diagnostics), diagnostics)

    def test_rejects_missing_checkpoint(self) -> None:
        diagnostics = valid_diagnostics()
        diagnostics["checkpoint_exists"] = False
        with self.assertRaisesRegex(RuntimeError, "checkpoint"):
            _validate_model_diagnostics(diagnostics)

    def test_rejects_preflight_without_model_action(self) -> None:
        diagnostics = valid_diagnostics()
        diagnostics["model_actions"] = 0
        with self.assertRaisesRegex(RuntimeError, "model decision"):
            _validate_model_diagnostics(diagnostics)

    def test_rejects_every_error_or_fallback_counter(self) -> None:
        for key in (
            "load_errors",
            "inference_errors",
            "illegal_model_actions",
            "fallback_actions",
            "illegal_fallback_actions",
            "emergency_legal_actions",
            "q_load_errors",
        ):
            with self.subTest(key=key):
                diagnostics = valid_diagnostics()
                diagnostics[key] = 1
                with self.assertRaisesRegex(RuntimeError, key):
                    _validate_model_diagnostics(diagnostics)


if __name__ == "__main__":
    unittest.main()
