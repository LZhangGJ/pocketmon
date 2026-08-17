from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def _iso(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode_row(episode: Any) -> dict[str, Any]:
    return {
        "id": int(episode.id),
        "create_time": _iso(episode.create_time),
        "end_time": _iso(episode.end_time),
        "state": _enum_name(episode.state),
        "type": _enum_name(episode.type),
        "agents": [
            {
                "submission_id": int(agent.submission_id),
                "index": int(agent.index),
                "reward": float(agent.reward),
                "state": _enum_name(agent.state),
                "team_name": str(agent.team_name),
                "team_id": int(agent.team_id),
            }
            for agent in (episode.agents or [])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download every Kaggle simulation replay attached to one submission."
    )
    parser.add_argument("submission_id", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    replay_dir = output / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    episodes = api.competition_list_episodes(args.submission_id)
    rows = [_episode_row(episode) for episode in episodes]
    manifest = {
        "submission_id": args.submission_id,
        "listed_at_utc": datetime.now(timezone.utc).isoformat(),
        "episode_count": len(rows),
        "episodes": rows,
    }
    manifest_path = output / "episodes.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for position, row in enumerate(rows, start=1):
        episode_id = row["id"]
        target = replay_dir / f"episode-{episode_id}-replay.json"
        if not target.is_file():
            api.competition_episode_replay(episode_id, str(replay_dir), quiet=True)
        row["replay_file"] = target.name
        row["replay_bytes"] = target.stat().st_size
        row["replay_sha256"] = _sha256(target)
        print(
            f"episodes={position}/{len(rows)} id={episode_id} bytes={target.stat().st_size}",
            flush=True,
        )

    manifest["downloaded_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "episode_count": len(rows)}))


if __name__ == "__main__":
    main()
