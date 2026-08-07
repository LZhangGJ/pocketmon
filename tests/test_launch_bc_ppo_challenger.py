from __future__ import annotations

import unittest

from scripts.launch_bc_ppo_challenger import validate_smoke_diagnostics


class BCToPPOHandoffTests(unittest.TestCase):
    def test_accepts_zero_error_diagnostics_for_both_seats(self) -> None:
        validate_smoke_diagnostics({"agent_diagnostics": [{}, {"model_actions": 2}]})

    def test_rejects_plural_error_counters_and_fallbacks(self) -> None:
        for field in ("load_errors", "inference_errors", "illegal_model_actions", "fallback_actions"):
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, field):
                validate_smoke_diagnostics({"agent_diagnostics": [{field: 1}, {}]})

    def test_requires_two_seat_diagnostics(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "both seats"):
            validate_smoke_diagnostics({"agent_diagnostics": [{}]})


if __name__ == "__main__":
    unittest.main()
