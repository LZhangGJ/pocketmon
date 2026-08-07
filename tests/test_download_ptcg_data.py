import unittest

from scripts.download_ptcg_data import resolve_daily_slug, select_episodes


class DownloadPtcgDataTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
