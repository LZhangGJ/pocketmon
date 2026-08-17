import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_ptcg_data import (
    download_dataset,
    resolve_daily_slug,
    resolve_direct_daily_slug,
    select_episodes,
)


class DownloadPtcgDataTests(unittest.TestCase):
    def test_resolve_direct_daily_slug(self):
        self.assertEqual(
            resolve_direct_daily_slug("2026-08-13"),
            ("2026-08-13", "kaggle/pokemon-tcg-ai-battle-episodes-2026-08-13"),
        )
        with self.assertRaises(ValueError):
            resolve_direct_daily_slug(None)

    def test_zero_selects_full_manifest(self):
        rows = [{"episode_id": str(index)} for index in range(3)]
        self.assertEqual(select_episodes(rows, 0), rows)
        self.assertEqual(select_episodes(rows, 2), rows[:2])

    def test_negative_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            select_episodes([], -1)

    def test_latest_index_row_is_selected(self):
        date, slug = resolve_daily_slug(None, [
            {"date": "2026-08-04", "daily_dataset_slug": "old"},
            {"date": "2026-08-05", "daily_dataset_slug": "kaggle/latest"},
        ])
        self.assertEqual(date, "2026-08-05")
        self.assertEqual(slug, "kaggle/latest")

    def test_bulk_download_uses_one_unzip_request(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)

            def fake_run(*args: str) -> None:
                (target / "manifest.csv").write_text("episode_id\n1\n", encoding="utf-8")

            with patch("scripts.download_ptcg_data.run_kaggle", side_effect=fake_run) as mocked:
                self.assertEqual(download_dataset("owner/daily", target), target / "manifest.csv")
            mocked.assert_called_once_with("datasets", "download", "owner/daily", "-p", str(target), "--unzip")


if __name__ == "__main__":
    unittest.main()
