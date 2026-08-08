from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "experiment7" / "integration"
if str(INTEGRATION) not in sys.path:
    sys.path.insert(0, str(INTEGRATION))

from run_reference_token_cache import (
    ORIGINAL_GUARD,
    adapt_reference_source,
    verify_reference_script,
)


class TokenCacheAdapterTests(unittest.TestCase):
    def test_reference_script_still_matches_package_manifest(self) -> None:
        script = verify_reference_script(ROOT / "experiment7" / "reference")
        self.assertEqual(script.name, "build_token_cache.py")

    def test_only_reference_action_guard_is_widened(self) -> None:
        source = "prefix\n" + ORIGINAL_GUARD + "suffix\n"
        adapted = adapt_reference_source(source, 128)
        self.assertEqual(
            adapted,
            source.replace("max_actions > 64", "max_actions > 128", 1),
        )

    def test_reference_guard_must_match_exactly_once(self) -> None:
        with self.assertRaises(RuntimeError):
            adapt_reference_source("", 128)
        with self.assertRaises(RuntimeError):
            adapt_reference_source(ORIGINAL_GUARD * 2, 128)

    def test_cap_cannot_narrow_reference_support(self) -> None:
        with self.assertRaises(ValueError):
            adapt_reference_source(ORIGINAL_GUARD, 63)


if __name__ == "__main__":
    unittest.main()
