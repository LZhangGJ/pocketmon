from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests


COMPETITION = "pokemon-tcg-ai-battle"
API_ROOT = "https://api.kaggle.com/v1"
WEB_ROOT = "https://www.kaggle.com"
ROOT = Path(__file__).resolve().parents[1]

# Public-score ordering observed on the competition Code page on 2026-08-06.
# The list is intentionally frozen so a rerun remains auditable even if the
# live leaderboard order changes later.
HIGH_SCORE_NOTEBOOKS = [
    ("biohack44/pok-mon-tcg-ai-battle-meta-snapshot-07-july", 947.5, 126),
    ("aristophanivan/probablity-v2", 933.8, 64),
    ("tetsutani/grimmsnarl-ex-damage-transfer-control", 909.3, 92),
    ("romanrozen/strong-start-baseline-agent-v10-lb-950", 892.4, 173),
    ("llccqq624/ptcg-meta-a-stable-submit", 864.6, 29),
    ("nursrijan/pokemon-ai-battle-agent-mega-lucario", 851.5, 49),
    ("lucifer19/battlecore-compact-agent", 846.8, 26),
    ("jazivxt/codex-sol-eclipse-alakazam", 840.3, 46),
    ("yaroslavkholmirzayev/mega-lucario-v2-crustle-aware-best-submit", 828.4, 9),
    ("makthanithin/improved-probabilistic-agent", 817.6, 6),
    ("biohack44/beating-the-day-1-1-crustle-bot-780c4c", 814.1, 8),
    ("rahuljiwane/pokemon-tcg-rahul-jiwane", 799.4, 18),
    ("prvsiyan/ptcg-ai-battle-visible-grim-belief-alakazam-v21", 798.3, 12),
    ("ravi123a321at/codex-sol-eclipse-alakazam-20a31c", 790.1, 6),
    ("skarin/phantom-dive-or-go-home-a-dragapult-ex-deck", 789.9, 44),
    ("daniilkrasnovvv/pokemon-conservative-probabilistic-agent", 784.8, 5),
    ("ravi123a321at/battlecore-compact-agent", 779.7, 3),
    ("biohack44/beating-the-day-2-new", 779.3, 8),
    ("prvsiyan/ptcg-ai-battle-search-audited-alakazam-v9", 778.2, 15),
    ("jazivxt/rising-tide-fixed-metal-v15-reproducible-agent", 774.0, 6),
]

METHOD_TOKENS = (
    "agent", "baseline", "rl", "mcts", "muzero", "search", "router",
    "deck", "imperfect", "ucb", "belief", "probabilistic", "control",
)


def request_json(
    session: requests.Session,
    service: str,
    method: str,
    payload: dict[str, Any],
    *,
    retries: int = 5,
) -> dict[str, Any]:
    url = f"{API_ROOT}/{service}/{method}"
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Failed POST {url} after {retries} attempts: {last_error}")


def topic_messages(messages: Iterable[dict[str, Any]], depth: int = 0) -> Iterable[tuple[int, dict[str, Any]]]:
    for message in messages:
        yield depth, message
        yield from topic_messages(message.get("replies") or [], depth + 1)


def write_topic_markdown(path: Path, topic: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    lines = [
        f"# {topic.get('title', '')}",
        "",
        f"- URL: {WEB_ROOT}{topic.get('topicUrl', '')}",
        f"- Topic ID: {topic.get('id', '')}",
        f"- Votes: {topic.get('votes', '')}",
        f"- Comments: {topic.get('commentCount', '')}",
        f"- Posted: {topic.get('postDate', '')}",
        "",
    ]
    for depth, message in topic_messages(messages):
        author = message.get("authorName") or message.get("author") or "unknown"
        posted = message.get("postDate") or ""
        lines.extend(
            [
                f"{'#' * min(6, depth + 2)} Message {message.get('id', '')} — {author}",
                "",
                f"Posted: {posted}",
                "",
                message.get("rawMarkdown") or message.get("content") or "",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def message_fingerprint(message: dict[str, Any]) -> str:
    payload = {
        "id": message.get("id"),
        "post_date": message.get("postDate"),
        "author": message.get("authorName") or message.get("author"),
        "content": message.get("rawMarkdown") or message.get("content") or "",
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def download_discussions(
    session: requests.Session,
    output_dir: Path,
    previous_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    service = "competitions.CompetitionApiService"
    first = request_json(
        session,
        service,
        "ListCompetitionTopics",
        {"competitionName": COMPETITION, "page": 1},
    )
    total = int(first.get("totalCount", 0))
    page_size = len(first.get("topics") or []) or 20
    pages = max(1, math.ceil(total / page_size))
    topics: list[dict[str, Any]] = []
    seen: set[int] = set()
    for page in range(1, pages + 1):
        response = first if page == 1 else request_json(
            session,
            service,
            "ListCompetitionTopics",
            {"competitionName": COMPETITION, "page": page},
        )
        for topic in response.get("topics") or []:
            topic_id = int(topic["id"])
            if topic_id not in seen:
                seen.add(topic_id)
                topics.append(topic)

    if len(topics) != total:
        raise RuntimeError(f"Topic completeness check failed: expected={total}, unique={len(topics)}")

    discussion_dir = output_dir / "discussions"
    raw_dir = discussion_dir / "raw"
    markdown_dir = discussion_dir / "markdown"
    raw_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    message_total = 0
    new_message_total = 0
    changed_topics = 0
    prior_topics = (previous_state or {}).get("topics", {})
    next_topics: dict[str, Any] = {}
    for index, topic in enumerate(topics, start=1):
        topic_id = int(topic["id"])
        response = request_json(
            session,
            service,
            "ListTopicMessages",
            {"competitionName": COMPETITION, "topicId": topic_id, "pageSize": -1},
        )
        messages = response.get("messages") or []
        flat_count = sum(1 for _ in topic_messages(messages))
        message_total += flat_count
        fingerprints = {
            str(message.get("id")): message_fingerprint(message)
            for _, message in topic_messages(messages)
        }
        prior_fingerprints = (prior_topics.get(str(topic_id)) or {}).get("messages", {})
        new_count = sum(prior_fingerprints.get(key) != value for key, value in fingerprints.items())
        topic_changed = new_count > 0 or str(topic_id) not in prior_topics
        new_message_total += new_count
        changed_topics += int(topic_changed)
        next_topics[str(topic_id)] = {
            "last_comment_post_date": topic.get("lastCommentPostDate", ""),
            "messages": fingerprints,
        }
        payload = {"topic": topic, "messages": messages}
        (raw_dir / f"{topic_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_topic_markdown(markdown_dir / f"{topic_id}.md", topic, messages)
        rows.append(
            {
                "id": topic_id,
                "title": topic.get("title", ""),
                "votes": topic.get("votes", 0),
                "comment_count_listed": topic.get("commentCount", 0),
                "messages_downloaded": flat_count,
                "new_or_changed_messages": new_count,
                "changed_since_previous": topic_changed,
                "post_date": topic.get("postDate", ""),
                "last_comment_post_date": topic.get("lastCommentPostDate", ""),
                "is_sticky": topic.get("isSticky", False),
                "url": f"{WEB_ROOT}{topic.get('topicUrl', '')}",
            }
        )
        if index % 20 == 0 or index == total:
            print(f"discussions={index}/{total} messages={message_total}", flush=True)

    with (discussion_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (discussion_dir / "topics.json").write_text(
        json.dumps(topics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return (
        {
            "topics": len(topics),
            "messages": message_total,
            "pages": pages,
            "changed_topics": changed_topics,
            "new_or_changed_messages": new_message_total,
        },
        {"topics": next_topics},
    )


def notebook_extension(metadata: dict[str, Any]) -> str:
    kernel_type = str(metadata.get("kernelType", "")).lower()
    language = str(metadata.get("language", "")).lower()
    if kernel_type == "notebook":
        return ".ipynb"
    return {"python": ".py", "r": ".R", "rmarkdown": ".Rmd", "julia": ".jl"}.get(language, ".txt")


def normalize_notebook_candidates(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [
            {
                "ref": ref,
                "public_score_observed": score,
                "votes_observed": votes,
                "discovery_sort": "frozen_public_score_2026-08-06",
            }
            for ref, score, votes in HIGH_SCORE_NOTEBOOKS
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("notebooks", []) if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError("notebook candidate file must contain a non-empty list")
    seen: set[str] = set()
    result = []
    for item in items:
        ref = str(item["ref"])
        if ref in seen:
            continue
        if ref.count("/") != 1:
            raise ValueError(f"invalid Kaggle notebook ref: {ref}")
        seen.add(ref)
        result.append(dict(item))
    return result


def kaggle_kernel_list(sort_by: str, page_size: int) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            sys.executable, "-m", "kaggle", "kernels", "list",
            "--competition", COMPETITION,
            "--sort-by", sort_by,
            "--page-size", str(page_size),
            "--format", "json",
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise ValueError(f"unexpected kernels list response for {sort_by}")
    return payload


def _run_time(item: dict[str, Any]) -> datetime | None:
    value = item.get("lastRunTime")
    if not value:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def select_live_notebook_candidates(
    score_items: list[dict[str, Any]],
    recent_items: list[dict[str, Any]],
    *,
    now: datetime,
    freshness_days: int = 21,
    limit: int = 24,
) -> list[dict[str, Any]]:
    """Mix fresh score-frontier notebooks with recent method-diverse agents."""
    if freshness_days <= 0 or limit <= 0:
        raise ValueError("freshness_days and limit must be positive")
    now = now.astimezone(timezone.utc)
    score_rank = {str(item["ref"]): rank for rank, item in enumerate(score_items, start=1)}
    recent_rank = {str(item["ref"]): rank for rank, item in enumerate(recent_items, start=1)}
    by_ref = {str(item["ref"]): item for item in score_items + recent_items}
    selected: list[str] = []

    def fresh(item: dict[str, Any]) -> bool:
        run_time = _run_time(item)
        return run_time is not None and 0 <= (now - run_time).total_seconds() <= freshness_days * 86400

    fresh_score_quota = max(1, int(round(limit * 2 / 3)))
    for item in score_items:
        ref = str(item["ref"])
        if fresh(item) and ref not in selected:
            selected.append(ref)
        if len(selected) >= fresh_score_quota:
            break

    for item in recent_items:
        ref = str(item["ref"])
        method_text = f"{ref} {item.get('title', '')}".lower()
        if fresh(item) and any(token in method_text for token in METHOD_TOKENS) and ref not in selected:
            selected.append(ref)
        if len(selected) >= limit:
            break

    for item in score_items:
        ref = str(item["ref"])
        if fresh(item) and ref not in selected:
            selected.append(ref)
        if len(selected) >= limit:
            break

    result: list[dict[str, Any]] = []
    for ref in selected:
        item = by_ref[ref]
        result.append({
            "ref": ref,
            "title_observed": item.get("title", ""),
            "votes_observed": item.get("totalVotes", ""),
            "last_run_time_observed": item.get("lastRunTime", ""),
            "score_rank_observed": score_rank.get(ref),
            "recent_rank_observed": recent_rank.get(ref),
            "discovery_sort": "fresh_score_frontier_plus_recent_method_diversity",
        })
    return result


def discover_live_notebook_candidates(freshness_days: int, limit: int) -> list[dict[str, Any]]:
    candidates = select_live_notebook_candidates(
        kaggle_kernel_list("scoreDescending", 100),
        kaggle_kernel_list("dateRun", 100),
        now=datetime.now(timezone.utc),
        freshness_days=freshness_days,
        limit=limit,
    )
    if not candidates:
        raise RuntimeError("live notebook discovery returned no fresh candidates")
    return candidates


def download_notebooks(
    session: requests.Session,
    output_dir: Path,
    candidates: list[dict[str, Any]],
    previous_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    service = "kernels.KernelsApiService"
    notebook_dir = output_dir / "notebooks"
    source_dir = notebook_dir / "source"
    metadata_dir = notebook_dir / "metadata"
    source_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    prior_notebooks = (previous_state or {}).get("notebooks", {})
    next_notebooks: dict[str, Any] = dict(prior_notebooks)
    changed_count = 0
    new_count = 0
    for rank, candidate in enumerate(candidates, start=1):
        ref = str(candidate["ref"])
        owner, slug = ref.split("/", 1)
        response = request_json(
            session,
            service,
            "GetKernel",
            {"userName": owner, "kernelSlug": slug},
        )
        metadata = response.get("metadata") or {}
        blob = response.get("blob") or {}
        source = blob.get("source") or ""
        source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
        previous = prior_notebooks.get(ref) or {}
        is_new = not previous
        changed = previous.get("source_sha256") != source_sha256
        new_count += int(is_new)
        changed_count += int(changed)
        extension = notebook_extension(metadata)
        stem = safe_name(ref.replace("/", "__"))
        source_path = source_dir / f"{rank:02d}__{stem}{extension}"
        metadata_path = metadata_dir / f"{rank:02d}__{stem}.json"
        source_path.write_text(source, encoding="utf-8")
        metadata_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(
            {
                "rank_observed": rank,
                "ref": ref,
                "title": metadata.get("title", ""),
                "public_score_observed": candidate.get("public_score_observed", ""),
                "votes_observed": candidate.get("votes_observed", metadata.get("totalVotes", "")),
                "updated_label_observed": candidate.get("updated_label_observed", ""),
                "last_run_time_observed": candidate.get("last_run_time_observed", ""),
                "score_rank_observed": candidate.get("score_rank_observed", ""),
                "recent_rank_observed": candidate.get("recent_rank_observed", ""),
                "discovery_sort": candidate.get("discovery_sort", "manual"),
                "last_run_time": metadata.get("lastRunTime", ""),
                "current_version_number": metadata.get("currentVersionNumber", ""),
                "source_sha256": source_sha256,
                "new_ref": is_new,
                "changed_since_previous": changed,
                "source_bytes": len(source.encode("utf-8")),
                "source_file": str(source_path.relative_to(output_dir)).replace("\\", "/"),
                "url": f"{WEB_ROOT}/code/{ref}",
            }
        )
        next_notebooks[ref] = {
            "last_run_time": metadata.get("lastRunTime", ""),
            "current_version_number": metadata.get("currentVersionNumber", ""),
            "source_sha256": source_sha256,
            "last_seen_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        print(f"notebooks={rank}/{len(candidates)} ref={ref} changed={changed}", flush=True)

    with (notebook_dir / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    numeric_scores = [
        float(row["public_score_observed"])
        for row in rows
        if row["public_score_observed"] not in (None, "")
    ]
    return (
        {
            "notebooks": len(rows),
            "new_refs": new_count,
            "changed_sources": changed_count,
            "score_min": min(numeric_scores) if numeric_scores else None,
            "score_max": max(numeric_scores) if numeric_scores else None,
        },
        {"notebooks": next_notebooks},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public Kaggle discussions and high-score notebooks")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research" / "kaggle_intelligence" / date.today().isoformat(),
    )
    parser.add_argument("--skip-discussions", action="store_true")
    parser.add_argument("--skip-notebooks", action="store_true")
    parser.add_argument(
        "--notebook-candidates",
        type=Path,
        help="JSON list discovered from the live Code page/API; preserves recency rather than using frozen global top score",
    )
    parser.add_argument("--notebook-freshness-days", type=int, default=21)
    parser.add_argument("--notebook-limit", type=int, default=24)
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "research" / "kaggle_intelligence" / "state.json",
        help="Persistent cursor/hash registry used to identify actual deltas",
    )
    args = parser.parse_args()

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "pocketmon-public-competition-research/1.0",
            "Content-Type": "application/json",
        }
    )

    state_path = args.state.resolve()
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}

    manifest: dict[str, Any] = {
        "competition": COMPETITION,
        "competition_url": f"{WEB_ROOT}/competitions/{COMPETITION}",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "public_only": True,
        "notebook_ranking_observed_on": date.today().isoformat(),
    }
    if not args.skip_discussions:
        manifest["discussions"], discussion_state = download_discussions(
            session, output_dir, state.get("discussions")
        )
        state["discussions"] = discussion_state
    if not args.skip_notebooks:
        candidates = (
            normalize_notebook_candidates(args.notebook_candidates)
            if args.notebook_candidates is not None
            else discover_live_notebook_candidates(args.notebook_freshness_days, args.notebook_limit)
        )
        manifest["notebooks"], notebook_state = download_notebooks(
            session, output_dir, candidates, state.get("notebooks")
        )
        state["notebooks"] = notebook_state

    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    (output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
