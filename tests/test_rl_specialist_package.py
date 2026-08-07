from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

from rl.agent_adapter import RLBCPolicyAdapter
from scripts.materialize_rl_specialist_agent import read_deck


ROOT = Path(__file__).resolve().parents[1]


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

    def test_submission_main_executes_without_dunder_file(self) -> None:
        """Mirror Kaggle's exec(main.py) loader, which does not define __file__."""
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "deck.csv").write_text(
                "".join(f"{card_id}\n" for card_id in range(60)), encoding="utf-8"
            )
            script = (
                "import os, pathlib\n"
                f"os.chdir({str(package)!r})\n"
                f"source = pathlib.Path({str(ROOT / 'agents' / 'rl_bc_specialist' / 'main.py')!r}).read_text(encoding='utf-8')\n"
                "namespace = {'__name__': 'submitted_agent'}\n"
                "exec(compile(source, '/kaggle_simulations/agent/main.py', 'exec'), namespace)\n"
                "assert namespace['PACKAGE_ROOT'] == pathlib.Path.cwd().resolve()\n"
                "assert len(namespace['agent']({'select': None})) == 60\n"
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(ROOT)
            subprocess.run(
                [sys.executable, "-c", script],
                check=True,
                cwd=package,
                env=environment,
                timeout=60,
            )


if __name__ == "__main__":
    unittest.main()
