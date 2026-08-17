from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INTEGRATION = ROOT / "experiment7" / "integration"


def agent(name: str, deck: str, policy: str, source_kind: str | None = None) -> dict:
    row = {
        "name": name,
        "deck_canonical_sha256": deck,
        "policyVersion": policy,
    }
    if source_kind is not None:
        row.update(
            {
                "sourceKind": source_kind,
                "immutable": True,
                "ppoUpdatesAllowed": False,
            }
        )
    return row


class PoolReconcilerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(INTEGRATION))
        cls.module = importlib.import_module("pool_reconciler")

    @classmethod
    def tearDownClass(cls) -> None:
        sys.path.remove(str(INTEGRATION))

    @staticmethod
    def fake_build(league: dict) -> dict:
        base = json.loads(Path(league["basePool"]["path"]).read_text(encoding="utf-8"))
        return {"agents": list(base["agents"]), "asyncLeague": {"basePool": {}}}

    def make_state(self, root: Path) -> tuple[Path, Path, Path]:
        base = root / "base.json"
        live = root / "live.json"
        league = root / "league.json"
        static = root / "static.json"
        large = root / "large.json"
        base.write_text(
            json.dumps({"agents": [agent("base", "deck-base", "policy-base")]}),
            encoding="utf-8",
        )
        live.write_text(json.dumps({"agents": []}), encoding="utf-8")
        league.write_text(
            json.dumps(
                {
                    "basePool": {"path": str(base)},
                    "poolPath": str(live),
                    "chains": {},
                }
            ),
            encoding="utf-8",
        )
        static.write_text(
            json.dumps(
                {"agents": [agent("static", "deck-static", "policy-static", "replay_static_bc")]}
            ),
            encoding="utf-8",
        )
        large.write_text(
            json.dumps(
                {
                    "agents": [
                        agent("large-b", "deck-b", "policy-large", "universal_bc_strict_incremental"),
                        agent("large-a", "deck-a", "policy-large", "universal_bc_strict_incremental"),
                    ]
                }
            ),
            encoding="utf-8",
        )
        return league, static, large

    def run_order(self, order: tuple[str, str]) -> tuple[list[str], list[str]]:
        with tempfile.TemporaryDirectory() as directory:
            league, static, large = self.make_state(Path(directory))
            paths = {
                "static": (static, "replay_static_bc"),
                "large": (large, "universal_bc_strict_incremental"),
            }
            for name in order:
                path, source_kind = paths[name]
                self.module.register_source_and_reconcile(
                    league, path, source_kind, build_live=self.fake_build
                )
            state = json.loads(league.read_text(encoding="utf-8"))
            base = json.loads(Path(state["basePool"]["path"]).read_text(encoding="utf-8"))
            live = json.loads(Path(state["poolPath"]).read_text(encoding="utf-8"))
            return [row["name"] for row in base["agents"]], [row["name"] for row in live["agents"]]

    def test_static_then_large_equals_large_then_static(self) -> None:
        self.assertEqual(self.run_order(("static", "large")), self.run_order(("large", "static")))
        base, live = self.run_order(("static", "large"))
        self.assertEqual(base, ["base", "static", "large-b", "large-a"])
        self.assertEqual(live, base)

    def test_concurrent_admissions_keep_both_canonical_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            league, static, large = self.make_state(Path(directory))
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(
                        self.module.register_source_and_reconcile,
                        league,
                        static,
                        "replay_static_bc",
                        build_live=self.fake_build,
                    ),
                    executor.submit(
                        self.module.register_source_and_reconcile,
                        league,
                        large,
                        "universal_bc_strict_incremental",
                        build_live=self.fake_build,
                    ),
                ]
                for future in futures:
                    future.result(timeout=10)
            state = json.loads(league.read_text(encoding="utf-8"))
            base = json.loads(Path(state["basePool"]["path"]).read_text(encoding="utf-8"))
            self.assertEqual(
                [row["name"] for row in base["agents"]],
                ["base", "static", "large-b", "large-a"],
            )
            self.assertEqual(len(state["canonicalPoolReconciler"]["sources"]), 3)


if __name__ == "__main__":
    unittest.main()
