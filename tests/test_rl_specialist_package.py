from pathlib import Path
import tempfile
import unittest

from rl.agent_adapter import RLBCPolicyAdapter
from scripts.materialize_rl_specialist_agent import read_deck


class RLSpecialistPackageTests(unittest.TestCase):
    def test_adapter_keeps_configured_deck_across_reset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            deck = list(range(60))
            adapter = RLBCPolicyAdapter(Path(directory) / "missing.pt", lambda _: deck, deck=deck)
            adapter.reset()
            self.assertEqual(adapter.diagnostics()["deck_cards"], 60)

    def test_adapter_rejects_bad_configured_deck(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "exactly 60"):
                RLBCPolicyAdapter(Path(directory) / "missing.pt", lambda _: [], deck=[1, 2])

    def test_read_deck_requires_sixty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deck.csv"
            path.write_text("1\n2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly 60"):
                read_deck(path)


if __name__ == "__main__":
    unittest.main()
