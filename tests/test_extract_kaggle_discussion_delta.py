from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.extract_kaggle_discussion_delta import build_delta, message_fingerprint


class ExtractKaggleDiscussionDeltaTest(unittest.TestCase):
    def test_extracts_new_edited_deleted_and_missing_topics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            snapshot = root / "snapshot"
            raw_dir = snapshot / "discussions" / "raw"
            raw_dir.mkdir(parents=True)

            old_message = {
                "id": 1,
                "postDate": "2026-08-07T00:00:00Z",
                "authorName": "alice",
                "rawMarkdown": "old",
                "replies": [],
            }
            deleted_message = {
                "id": 2,
                "postDate": "2026-08-07T01:00:00Z",
                "authorName": "bob",
                "rawMarkdown": "deleted",
                "replies": [],
            }
            unchanged_message = {
                "id": 4,
                "postDate": "2026-08-07T03:00:00Z",
                "authorName": "dora",
                "rawMarkdown": "same",
                "replies": [],
            }
            previous_state = {
                "discussions": {
                    "topics": {
                        "10": {
                            "messages": {
                                "1": message_fingerprint(old_message),
                                "2": message_fingerprint(deleted_message),
                            }
                        },
                        "20": {"messages": {"4": message_fingerprint(unchanged_message)}},
                        "99": {"messages": {"9": "previous-fingerprint"}},
                    }
                }
            }
            previous_state_path = root / "previous_state.json"
            previous_state_path.write_text(json.dumps(previous_state), encoding="utf-8")

            manifest = {
                "competition": "pokemon-tcg-ai-battle",
                "downloaded_at_utc": "2026-08-08T00:00:00+00:00",
                "public_only": True,
                "discussions": {"topics": 2, "messages": 3, "pages": 1},
            }
            (snapshot / "download_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            edited_message = dict(old_message, rawMarkdown="edited")
            nested_message = {
                "id": 3,
                "postDate": "2026-08-08T00:30:00Z",
                "authorName": "carol",
                "rawMarkdown": "new reply",
                "replies": [],
            }
            root_message = {
                "id": 5,
                "postDate": "2026-08-08T00:00:00Z",
                "authorName": "root",
                "rawMarkdown": "new root",
                "replies": [nested_message],
            }
            topic_10 = {
                "id": 10,
                "title": "Topic ten",
                "topicUrl": "/competitions/pokemon-tcg-ai-battle/discussion/10",
                "postDate": "2026-08-01",
                "lastCommentPostDate": "2026-08-08",
                "votes": 5,
            }
            topic_20 = {
                "id": 20,
                "title": "Topic twenty",
                "topicUrl": "/competitions/pokemon-tcg-ai-battle/discussion/20",
                "postDate": "2026-08-01",
                "lastCommentPostDate": "2026-08-07",
                "votes": 2,
            }
            (raw_dir / "10.json").write_text(
                json.dumps({"topic": topic_10, "messages": [edited_message, root_message]}),
                encoding="utf-8",
            )
            (raw_dir / "20.json").write_text(
                json.dumps({"topic": topic_20, "messages": [unchanged_message]}),
                encoding="utf-8",
            )

            output = root / "delta" / "run-1"
            summary = build_delta(snapshot, previous_state_path, output)

            self.assertEqual(summary["delta"]["changed_topics"], 1)
            self.assertEqual(summary["delta"]["new_messages"], 2)
            self.assertEqual(summary["delta"]["edited_messages"], 1)
            self.assertEqual(summary["delta"]["deleted_messages"], 1)
            self.assertEqual(summary["delta"]["missing_topic_ids"], ["99"])

            records = [
                json.loads(line)
                for line in (output / "messages.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                sorted(record["status"] for record in records),
                ["deleted", "edited", "new", "new"],
            )
            self.assertFalse((output / "raw" / "20.json").exists())
            self.assertTrue((output / "markdown" / "10.md").is_file())
            self.assertIn("Topic ten", (output / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
