from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiment7.integration.agent_isolation import (
    call_agent,
    isolated_agent_workdir,
    load_agent,
)


class CollectUniversalPpoRolloutsTests(unittest.TestCase):
    def test_import_time_deck_write_is_confined_to_disposable_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            agent = Path(temporary) / "agent"
            agent.mkdir()
            frozen_deck = "".join(f"{card}\n" for card in range(60))
            (agent / "deck.csv").write_text(frozen_deck, encoding="utf-8")
            (agent / "main.py").write_text(
                "from pathlib import Path\n"
                "Path('deck.csv').write_text('999\\n', encoding='utf-8')\n"
                "def agent(observation):\n"
                "    Path('runtime.txt').write_text('isolated', encoding='utf-8')\n"
                "    return []\n",
                encoding="utf-8",
            )

            with isolated_agent_workdir(agent) as workdir:
                module = load_agent(agent, "test_import_time_deck_writer", workdir)
                call_agent(module, {}, workdir)
                self.assertEqual((workdir / "deck.csv").read_text(encoding="utf-8"), "999\n")
                self.assertEqual((workdir / "runtime.txt").read_text(encoding="utf-8"), "isolated")

            self.assertEqual((agent / "deck.csv").read_text(encoding="utf-8"), frozen_deck)
            self.assertFalse((agent / "runtime.txt").exists())


if __name__ == "__main__":
    unittest.main()
