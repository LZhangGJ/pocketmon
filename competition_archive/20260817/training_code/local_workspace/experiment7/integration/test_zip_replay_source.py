from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_from_pocketmon_replays import close_replay_archives, read_replay_bytes
from build_replay_catalog import ZipReplaySource, _manifest_metadata


class ZipReplaySourceTest(unittest.TestCase):
    def test_streams_member_and_manifest_without_extracting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "2026-08-15.zip"
            raw = b'{"episode_id": 123}'
            with zipfile.ZipFile(archive_path, "w") as output:
                output.writestr("123.json", raw)
                output.writestr(
                    "manifest.csv",
                    "episode_id,create_time,min_score\n123,2026-08-15T00:00:00Z,1000.1\n",
                )
            with zipfile.ZipFile(archive_path) as archive:
                source = ZipReplaySource(
                    archive_path,
                    archive,
                    "123.json",
                    source_date="2026-08-15",
                    mtime=1.0,
                )
                self.assertEqual(source.read_bytes(), raw)
                self.assertEqual(source.stem, "123")
                self.assertEqual(_manifest_metadata(archive_path, archive)[123]["min_score"], "1000.1")
                locator = source.locator
            try:
                self.assertEqual(read_replay_bytes(locator), raw)
            finally:
                close_replay_archives()


if __name__ == "__main__":
    unittest.main()
