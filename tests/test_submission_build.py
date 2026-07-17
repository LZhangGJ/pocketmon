from __future__ import annotations

import py_compile
import unittest
from pathlib import Path

from scripts.build_submission import read_deck


ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agents" / "lucario_rule"


class SubmissionSourceTests(unittest.TestCase):
    def test_deck_has_sixty_cards(self) -> None:
        self.assertEqual(len(read_deck(AGENT / "deck.csv")), 60)

    def test_agent_compiles(self) -> None:
        py_compile.compile(str(AGENT / "main.py"), doraise=True)


if __name__ == "__main__":
    unittest.main()
