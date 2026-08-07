from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.submission_ledger import record_submission, sha256


class SubmissionLedgerTests(unittest.TestCase):
    def test_records_and_updates_by_date_and_archive_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "agent.tar.gz"
            archive.write_bytes(b"archive")
            ledger = root / "ledger.json"
            base = {
                "date_jst": "2026-08-07",
                "archive": str(archive),
                "archive_sha256": sha256(archive),
                "submission_ref": "1",
                "status": "PENDING",
            }
            record_submission(ledger, base)
            record_submission(ledger, {**base, "status": "COMPLETE", "public_score": 561.6})
            payload = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["submissions"]), 1)
            self.assertEqual(payload["submissions"][0]["status"], "COMPLETE")
            self.assertEqual(payload["submissions"][0]["public_score"], 561.6)

    def test_rejects_wrong_archive_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "agent.tar.gz"
            archive.write_bytes(b"archive")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                record_submission(
                    root / "ledger.json",
                    {
                        "date_jst": "2026-08-07",
                        "archive": str(archive),
                        "archive_sha256": "0" * 64,
                    },
                )


if __name__ == "__main__":
    unittest.main()
