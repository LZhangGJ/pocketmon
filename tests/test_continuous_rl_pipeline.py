from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from unittest.mock import patch

from scripts.continuous_rl_pipeline import choose_trainer, wait_for_files


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

    def test_trainer_selection_prefers_idle_gpu_over_larger_busy_gpu(self) -> None:
        readings = {
            "idle": (20_000, 24_000, 0, 1),
            "busy": (70_000, 80_000, 92, 3),
        }
        with patch("scripts.continuous_rl_pipeline.free_gpu", side_effect=lambda host, _: readings[host]):
            host, gpu, audit = choose_trainer({
                "trainer_hosts": ["busy", "idle"],
                "local_host": "coordinator",
            })
        self.assertEqual((host, gpu), ("idle", 1))
        self.assertEqual(audit["busy"]["utilization_percent"], 92)


if __name__ == "__main__":
    unittest.main()
