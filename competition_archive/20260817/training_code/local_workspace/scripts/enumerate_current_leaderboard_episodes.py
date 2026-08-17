from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


SNAPSHOT = Path(".tmp_current_leaderboard/current_snapshot.json")
OUTPUT = Path(".tmp_current_leaderboard/current_episode_index.json")


def enum_name(value):
    return str(getattr(value, "name", value))


def iso(value):
    return value.isoformat() if hasattr(value, "isoformat") else str(value or "")


def save(payload):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, OUTPUT)


def main() -> None:
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN is not set")
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    submissions = {}
    for team in snapshot["selected"]:
        if team.get("mapping_status") != "ok":
            raise RuntimeError(f"incomplete team mapping: {team['team_id']}")
        for submission in team["active_submissions"]:
            submissions[int(submission["submission_id"])] = {
                "team_id": int(team["team_id"]),
                "team_name": team["team_name"],
                "leaderboard_score": float(team["score"]),
                "submission_public_score": submission.get("public_score"),
            }
    payload = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "leaderboard_checked_at": snapshot["checked_at"],
        "qualifying_team_count": snapshot["selected_count"],
        "active_submission_count": len(submissions),
        "processed_submissions": [],
        "errors": [],
        "episodes": {},
    }
    api = KaggleApi()
    api.authenticate()
    for position, (submission_id, owner) in enumerate(submissions.items(), 1):
        episodes = None
        error = None
        for delay in (0, 2, 5, 10, 20, 40):
            if delay:
                time.sleep(delay)
            try:
                episodes = api.competition_list_episodes(submission_id) or []
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if episodes is None:
            payload["errors"].append({"submission_id": submission_id, "error": error})
        else:
            payload["processed_submissions"].append(submission_id)
            for episode in episodes:
                episode_id = str(int(episode.id))
                row = payload["episodes"].setdefault(
                    episode_id,
                    {
                        "episode_id": int(episode.id),
                        "create_time": iso(episode.create_time),
                        "end_time": iso(episode.end_time),
                        "state": enum_name(episode.state),
                        "type": enum_name(episode.type),
                        "agents": [
                            {
                                "submission_id": int(agent.submission_id),
                                "index": int(agent.index),
                                "reward": float(agent.reward),
                                "state": enum_name(agent.state),
                                "team_name": str(agent.team_name),
                                "team_id": int(agent.team_id),
                            }
                            for agent in (episode.agents or [])
                        ],
                        "qualifying_sources": [],
                    },
                )
                source = {"submission_id": submission_id, **owner}
                if source not in row["qualifying_sources"]:
                    row["qualifying_sources"].append(source)
        if position % 10 == 0:
            save(payload)
            print(json.dumps({"processed": position, "total": len(submissions), "episodes": len(payload["episodes"]), "errors": len(payload["errors"])}), flush=True)
        time.sleep(1.0)
    save(payload)
    print(json.dumps({"processed": len(payload["processed_submissions"]), "total": len(submissions), "episodes": len(payload["episodes"]), "errors": len(payload["errors"])}))
    if payload["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
