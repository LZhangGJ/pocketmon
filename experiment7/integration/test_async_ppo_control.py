from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


INTEGRATION = Path(__file__).resolve().parent
if str(INTEGRATION) not in sys.path:
    sys.path.insert(0, str(INTEGRATION))

from async_ppo_control import initialize, publish_snapshot, read_json  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class AsyncPpoControlTest(unittest.TestCase):
    def test_three_live_snapshots_are_added_to_canonical_pool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.json"
            write_json(
                base,
                {
                    "agents": [
                        {"name": "a03", "archetype": "A03"},
                        {"name": "alakazam", "archetype": "alakazam"},
                    ]
                },
            )
            chains = {}
            for index, (name, archetype) in enumerate(
                (("a05", "A05"), ("lucario", "LUCARIO"), ("a08", "A08")), start=1
            ):
                checkpoint = root / f"{name}.pt"
                checkpoint.write_bytes(f"checkpoint-{name}".encode())
                deck = root / f"{name}.csv"
                deck.write_text("1\n" * 60, encoding="utf-8")
                chains[name] = {
                    "deckName": name,
                    "archetypeId": archetype,
                    "archetypeLabel": name,
                    "deckPath": str(deck),
                    "deckSha256": f"deck-{index}",
                    "teacher": str(checkpoint),
                    "current": {"generation": 7, "checkpoint": str(checkpoint)},
                }
            config = root / "config.json"
            league_path = root / "league.json"
            pool_path = root / "pool.json"
            write_json(
                config,
                {
                    "basePool": {"path": str(base)},
                    "poolPath": str(pool_path),
                    "referenceRoot": str(root),
                    "engineCatalog": str(root / "engine.json"),
                    "cgDir": str(root),
                    "sources": str(root / "sources.json"),
                    "chains": chains,
                },
            )
            initialize(league_path, config)
            self.assertEqual(len(read_json(pool_path)["agents"]), 2)
            for name, chain in chains.items():
                agent = root / f"agent-{name}"
                agent.mkdir()
                (agent / "main.py").write_text("pass\n", encoding="utf-8")
                (agent / "deck.csv").write_text("1\n" * 60, encoding="utf-8")
                manifest = root / f"manifest-{name}.json"
                write_json(
                    manifest,
                    {
                        "packages": [
                            {
                                "name": f"live-{name}",
                                "agentDir": str(agent),
                                "deckSha256": chain["deckSha256"],
                                "archetypeId": chain["archetypeId"],
                                "directorySha256": f"directory-{name}",
                            }
                        ]
                    },
                )
                publish_snapshot(
                    league_path,
                    name,
                    7,
                    Path(chain["current"]["checkpoint"]),
                    manifest,
                )
            pool = read_json(pool_path)
            self.assertEqual(len(pool["agents"]), 5)
            self.assertEqual(
                [row["canonical_archetype"] for row in pool["agents"][:2]],
                ["A03", "A03"],
            )
            self.assertEqual(len(pool["asyncLeague"]["dynamicAgents"]), 3)


if __name__ == "__main__":
    unittest.main()
