from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from scripts.continuous_rl_pipeline import wait_for_files


class ContinuousPipelineTests(unittest.TestCase):
    def test_waits_for_delayed_shared_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "remote-result.json"

            def create() -> None:
                time.sleep(0.05)
                path.write_text("{}", encoding="utf-8")

            worker = threading.Thread(target=create)
            worker.start()
            wait_for_files([path], timeout_seconds=2.0)
            worker.join()
            self.assertTrue(path.is_file())

    def test_reports_missing_shared_artifact_after_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing"
            with self.assertRaisesRegex(RuntimeError, "did not become visible"):
                wait_for_files([path], timeout_seconds=0.01)


if __name__ == "__main__":
    unittest.main()
