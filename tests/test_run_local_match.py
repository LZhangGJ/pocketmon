import sys
import tempfile
import types
import unittest
from pathlib import Path

from experiment7.integration.agent_isolation import isolated_agent_workdir
from scripts.run_local_match import load_agent


class RunLocalMatchTests(unittest.TestCase):
    def _write_agent(self, root: Path, name: str, version: str) -> Path:
        agent = root / name
        package = agent / "rl"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "agent_adapter.py").write_text(
            f"VERSION = {version!r}\n", encoding="utf-8"
        )
        (agent / "main.py").write_text(
            "from rl.agent_adapter import VERSION\n", encoding="utf-8"
        )
        return agent

    def tearDown(self) -> None:
        for module_name in list(sys.modules):
            if module_name == "rl" or module_name.startswith("rl."):
                del sys.modules[module_name]

    def test_import_time_deck_write_uses_disposable_workdir(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            agent = root / "writer"
            agent.mkdir()
            frozen = "".join(f"{card}\n" for card in range(60))
            (agent / "deck.csv").write_text(frozen, encoding="utf-8")
            (agent / "main.py").write_text(
                "from pathlib import Path\n"
                "Path('deck.csv').write_text('999\\n', encoding='utf-8')\n"
                "def agent(observation): return []\n",
                encoding="utf-8",
            )

            with isolated_agent_workdir(agent) as workdir:
                load_agent(agent, "isolated_import_writer", workdir)
                self.assertEqual((workdir / "deck.csv").read_text(encoding="utf-8"), "999\n")

            self.assertEqual((agent / "deck.csv").read_text(encoding="utf-8"), frozen)

    def test_self_contained_agent_packages_are_isolated_in_both_orders(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = self._write_agent(root, "old", "old")
            new = self._write_agent(root, "new", "new")

            old_module = load_agent(old, "isolation_old_first")
            new_module = load_agent(new, "isolation_new_second")
            self.assertEqual(old_module.VERSION, "old")
            self.assertEqual(new_module.VERSION, "new")

            new_module = load_agent(new, "isolation_new_first")
            old_module = load_agent(old, "isolation_old_second")
            self.assertEqual(new_module.VERSION, "new")
            self.assertEqual(old_module.VERSION, "old")

    def test_bundled_cg_does_not_remove_installed_engine(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            agent = self._write_agent(root, "submission", "submission")
            bundled_cg = agent / "cg"
            bundled_cg.mkdir()
            (bundled_cg / "__init__.py").write_text("", encoding="utf-8")
            engine = types.ModuleType("cg")
            engine.marker = "installed-engine"
            sys.modules["cg"] = engine

            load_agent(agent, "submission_with_cg")
            self.assertIs(sys.modules["cg"], engine)


if __name__ == "__main__":
    unittest.main()
