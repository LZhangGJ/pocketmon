#!/usr/bin/env python3
"""Extract durable message-level deltas from a full Kaggle discussion snapshot.

The downloader intentionally writes a complete public snapshot so every run can
be audited.  This helper compares that snapshot with the state cursor captured
before the download and writes only new, edited, or removed messages into the
repository.  The full snapshot can then be retained as a short-lived workflow
artifact without growing Git history by tens of megabytes per day.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


WEB_ROOT = "https://www.kaggle.com"


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def message_fingerprint(message: dict[str, Any]) -> str:
    """Match the fingerprint used by download_kaggle_intelligence.py exactly."""
    payload = {
        "id": message.get("id"),
        "post_date": message.get("postDate"),
        "author": message.get("authorName") or message.get("author"),
        "content": message.get("rawMarkdown") or message.get("content") or "",
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def iter_messages(
    messages: list[dict[str, Any]],
    *,
    depth: int = 0,
    parent_id: str | None = None,
) -> Iterator[tuple[int, str | None, dict[str, Any]]]:
    for message in messages:
        yield depth, parent_id, message
        current_id = str(message.get("id") or "") or parent_id
        replies = message.get("replies") or []
        if isinstance(replies, list):
            yield from iter_messages(replies, depth=depth + 1, parent_id=current_id)


def topic_url(topic: dict[str, Any], topic_id: str) -> str:
    relative = str(topic.get("topicUrl") or "")
    if relative.startswith("http://") or relative.startswith("https://"):
        return relative
    if relative:
        return f"{WEB_ROOT}{relative}"
    return f"{WEB_ROOT}/competitions/pokemon-tcg-ai-battle/discussion/{topic_id}"


def markdown_escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def write_topic_markdown(path: Path, row: dict[str, Any], records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    title = row["title"] or f"Topic {row['topic_id']}"
    lines = [
        f"# {title}",
        "",
        f"- URL: {row['url']}",
        f"- Topic ID: {row['topic_id']}",
        f"- Last comment: {row['last_comment_post_date']}",
        f"- Delta: {row['new_messages']} new, {row['edited_messages']} edited, "
        f"{row['deleted_messages']} deleted",
        "",
    ]
    for record in records:
        status = str(record["status"]).upper()
        message_id = record["message_id"]
        author = record.get("author") or "unknown"
        heading_level = min(6, 2 + int(record.get("depth") or 0))
        lines.extend(
            [
                f"{'#' * heading_level} {status} message {message_id} — {author}",
                "",
                f"Posted: {record.get('post_date') or ''}",
                "",
            ]
        )
        if status == "DELETED":
            lines.extend(["The message was present in the previous cursor but is absent now.", ""])
        else:
            lines.extend([record.get("content") or "", ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_readme(path: Path, summary: dict[str, Any], topics: list[dict[str, Any]]) -> None:
    delta = summary["delta"]
    snapshot = summary["snapshot"]
    lines = [
        f"# Kaggle discussion delta — {summary['run_label']}",
        "",
        f"- Downloaded at (UTC): {summary.get('source_downloaded_at_utc') or ''}",
        f"- Full snapshot: {snapshot.get('topics', 0)} topics / {snapshot.get('messages', 0)} messages",
        f"- Changed topics: {delta['changed_topics']}",
        f"- Messages: {delta['new_messages']} new / {delta['edited_messages']} edited / "
        f"{delta['deleted_messages']} deleted",
        f"- Topics absent from this snapshot: {delta['missing_topics']}",
        "",
        "The full snapshot is retained as a GitHub Actions artifact. This directory stores only durable deltas.",
        "",
    ]
    if topics:
        lines.extend(
            [
                "| Topic | New | Edited | Deleted | Last comment |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in topics:
            title = markdown_escape(row["title"] or row["topic_id"])
            lines.append(
                f"| [{title}]({row['url']}) | {row['new_messages']} | "
                f"{row['edited_messages']} | {row['deleted_messages']} | "
                f"{markdown_escape(row['last_comment_post_date'])} |"
            )
    else:
        lines.append("No public discussion message changed relative to the previous cursor.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_delta(snapshot_dir: Path, previous_state_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = snapshot_dir / "download_manifest.json"
    raw_dir = snapshot_dir / "discussions" / "raw"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing download manifest: {manifest_path}")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"missing discussion raw directory: {raw_dir}")

    manifest = read_json(manifest_path, {})
    previous_state = read_json(previous_state_path, {})
    previous_topics = ((previous_state.get("discussions") or {}).get("topics") or {})
    if not isinstance(previous_topics, dict):
        raise ValueError("previous state discussions.topics must be an object")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    topic_rows: list[dict[str, Any]] = []
    current_topic_ids: set[str] = set()
    invalid_message_ids = 0

    for raw_path in sorted(raw_dir.glob("*.json"), key=lambda path: path.stem):
        payload = read_json(raw_path, {})
        topic = payload.get("topic") or {}
        messages = payload.get("messages") or []
        topic_id = str(topic.get("id") or raw_path.stem)
        current_topic_ids.add(topic_id)
        prior_fingerprints = ((previous_topics.get(topic_id) or {}).get("messages") or {})
        if not isinstance(prior_fingerprints, dict):
            prior_fingerprints = {}

        current_message_ids: set[str] = set()
        records: list[dict[str, Any]] = []
        for depth, parent_id, message in iter_messages(messages):
            message_id = str(message.get("id") or "")
            if not message_id:
                invalid_message_ids += 1
                continue
            current_message_ids.add(message_id)
            fingerprint = message_fingerprint(message)
            previous_fingerprint = prior_fingerprints.get(message_id)
            if previous_fingerprint == fingerprint:
                continue
            status = "new" if previous_fingerprint is None else "edited"
            record = {
                "topic_id": topic_id,
                "topic_title": topic.get("title") or "",
                "topic_url": topic_url(topic, topic_id),
                "message_id": message_id,
                "status": status,
                "depth": depth,
                "parent_id": parent_id,
                "author": message.get("authorName") or message.get("author") or "unknown",
                "post_date": message.get("postDate") or "",
                "content": message.get("rawMarkdown") or message.get("content") or "",
                "fingerprint": fingerprint,
            }
            records.append(record)

        for message_id in sorted(set(prior_fingerprints) - current_message_ids):
            records.append(
                {
                    "topic_id": topic_id,
                    "topic_title": topic.get("title") or "",
                    "topic_url": topic_url(topic, topic_id),
                    "message_id": message_id,
                    "status": "deleted",
                    "depth": 0,
                    "parent_id": None,
                    "author": "",
                    "post_date": "",
                    "content": "",
                    "fingerprint": "",
                }
            )

        if not records:
            continue
        records.sort(key=lambda row: (row.get("post_date") or "", row["message_id"], row["status"]))
        counts = {
            status: sum(1 for record in records if record["status"] == status)
            for status in ("new", "edited", "deleted")
        }
        row = {
            "topic_id": topic_id,
            "title": topic.get("title") or "",
            "url": topic_url(topic, topic_id),
            "post_date": topic.get("postDate") or "",
            "last_comment_post_date": topic.get("lastCommentPostDate") or "",
            "votes": topic.get("votes") or 0,
            "new_messages": counts["new"],
            "edited_messages": counts["edited"],
            "deleted_messages": counts["deleted"],
        }
        topic_rows.append(row)
        all_records.extend(records)
        (output_dir / "raw").mkdir(parents=True, exist_ok=True)
        (output_dir / "raw" / f"{topic_id}.json").write_text(
            json.dumps({"topic": row, "messages": records}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        write_topic_markdown(output_dir / "markdown" / f"{topic_id}.md", row, records)

    topic_rows.sort(
        key=lambda row: (row.get("last_comment_post_date") or "", row["topic_id"]),
        reverse=True,
    )
    all_records.sort(
        key=lambda row: (row.get("post_date") or "", row["topic_id"], row["message_id"])
    )

    missing_topic_ids = sorted(set(previous_topics) - current_topic_ids)
    discussions = manifest.get("discussions") or {}
    summary = {
        "competition": manifest.get("competition") or "pokemon-tcg-ai-battle",
        "public_only": bool(manifest.get("public_only", True)),
        "run_label": output_dir.name,
        "source_downloaded_at_utc": manifest.get("downloaded_at_utc") or "",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "previous_state_present": previous_state_path.is_file(),
        "snapshot": {
            "topics": int(discussions.get("topics") or len(current_topic_ids)),
            "messages": int(discussions.get("messages") or 0),
            "pages": int(discussions.get("pages") or 0),
        },
        "delta": {
            "changed_topics": len(topic_rows),
            "new_messages": sum(row["new_messages"] for row in topic_rows),
            "edited_messages": sum(row["edited_messages"] for row in topic_rows),
            "deleted_messages": sum(row["deleted_messages"] for row in topic_rows),
            "missing_topics": len(missing_topic_ids),
            "missing_topic_ids": missing_topic_ids,
            "invalid_message_ids": invalid_message_ids,
        },
    }

    with (output_dir / "topics.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "topic_id",
            "title",
            "url",
            "post_date",
            "last_comment_post_date",
            "votes",
            "new_messages",
            "edited_messages",
            "deleted_messages",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(topic_rows)

    with (output_dir / "messages.jsonl").open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_readme(output_dir / "README.md", summary, topic_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--previous-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_delta(
        args.snapshot.resolve(),
        args.previous_state.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
