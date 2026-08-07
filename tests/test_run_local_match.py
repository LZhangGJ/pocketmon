import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
