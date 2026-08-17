import base64
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.materialize_notebook_agent import materialize


class MaterializeNotebookAgentTests(unittest.TestCase):
    def test_extracts_one_agent_from_embedded_tar(self):
        archive_bytes = io.BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w:gz") as archive:
            for name, content in {
                "submission/main.py": b"def agent(observation):\n    return []\n",
                "submission/deck.csv": ("1\n" * 60).encode(),
            }.items():
                info = tarfile.TarInfo(name)
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        encoded = base64.b64encode(archive_bytes.getvalue()).decode()
        payload = {
            "cells": [{"cell_type": "code", "source": f"AGENT_B64 = '{encoded}'\n"}]
        }
        with tempfile.TemporaryDirectory() as temp:
            notebook = Path(temp) / "agent.ipynb"
            output = Path(temp) / "agent"
            notebook.write_text(json.dumps(payload), encoding="utf-8")
            materialize(notebook, output)
            self.assertTrue((output / "main.py").is_file())
            self.assertEqual(len((output / "deck.csv").read_text().splitlines()), 60)


if __name__ == "__main__":
    unittest.main()
