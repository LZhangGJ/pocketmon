from __future__ import annotations

import json
import py_compile
import tempfile
import unittest
from pathlib import Path

from scripts.build_submission import read_deck
from scripts.materialize_notebook_agent import extract_commented_deck, extract_literal_deck, materialize


ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agents" / "lucario_rule"


class SubmissionSourceTests(unittest.TestCase):
    def test_deck_has_sixty_cards(self) -> None:
        self.assertEqual(len(read_deck(AGENT / "deck.csv")), 60)

    def test_agent_compiles(self) -> None:
        py_compile.compile(str(AGENT / "main.py"), doraise=True)

    def test_matchup_constants_are_present(self) -> None:
        source = (AGENT / "main.py").read_text(encoding="utf-8")
        for matchup in ("alakazam", "archaludon", "library_out", "starmie"):
            self.assertIn(f'"{matchup}"', source)

    def test_materializer_uses_fallback_deck(self) -> None:
        notebook = {
            "cells": [
                {"cell_type": "code", "source": ["%%writefile main.py\n", "def agent(obs):\n", "    return []\n"]}
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            notebook_path = temp / "agent.ipynb"
            notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
            output = temp / "out"
            written = materialize(notebook_path, output, AGENT / "deck.csv")
            self.assertEqual({path.name for path in written}, {"main.py", "deck.csv"})
            self.assertEqual(len(read_deck(output / "deck.csv")), 60)

    def test_extracts_literal_deck_without_execution(self) -> None:
        cards = list(range(60))
        payload = {"cells": [{"cell_type": "code", "source": [f"DECK = {cards!r}\n"]}]}
        self.assertEqual(extract_literal_deck(payload), cards)

    def test_extracts_commented_deck(self) -> None:
        payload = {
            "cells": [
                {"cell_type": "code", "source": ["CardA = 7  # ×30\n", "CardB = 8  # x30\n"]}
            ]
        }
        self.assertEqual(extract_commented_deck(payload), [7] * 30 + [8] * 30)


if __name__ == "__main__":
    unittest.main()
