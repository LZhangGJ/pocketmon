from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from scripts.graceful_restart_after_pid import process_matches, validate_stop_file


class GracefulRestartTests(unittest.TestCase):
    def test_current_process_matches_python_command(self) -> None:
        self.assertTrue(process_matches(os.getpid(), "python"))

    def test_stop_marker_must_be_named_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "STOP"
            self.assertEqual(validate_stop_file(marker), marker.resolve())
            with self.assertRaisesRegex(ValueError, "named STOP"):
                validate_stop_file(Path(directory) / "anything_else")


if __name__ == "__main__":
    unittest.main()
