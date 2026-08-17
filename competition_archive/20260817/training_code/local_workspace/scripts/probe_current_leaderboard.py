from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import ApiGetLeaderboardRequest, KaggleApi


COMPETITION = "pokemon-tcg-ai-battle"
OUTPUT = Path(".tmp_current_leaderboard/current_snapshot.json")


def field(obj, *names):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def main() -> None:
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise SystemExit("KAGGLE_API_TOKEN is not set")
    api = KaggleApi()
    api.authenticate()
    entries = []
    page_token = None
    with api.build_kaggle_client() as kaggle:
        while True:
            request = ApiGetLeaderboardRequest()
            request.competition_name = COMPETITION
            request.page_size = 500
            request.page_token = page_token
            response = kaggle.competitions.competition_api_client.get_leaderboard(request)
            page = list(response.submissions or [])
            entries.extend(page)
            if not response.next_page_token or not page:
                break
            if float(field(page[-1], "score") or 0.0) <= 1000.0:
                break
            page_token = response.next_page_token
    selected = []
    for entry in entries:
        score = float(field(entry, "score") or 0.0)
        if score <= 1000.0:
            continue
        team_id = int(field(entry, "team_id", "teamId"))
        selected.append(
            {
                "team_id": team_id,
                "team_name": str(field(entry, "team_name", "teamName") or ""),
                "score": score,
                "submission_date": str(field(entry, "submission_date", "submissionDate") or ""),
                "active_submissions": [],
                "mapping_status": "pending",
            }
        )
    payload = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "competition": COMPETITION,
        "leaderboard_rows_returned": len(entries),
        "threshold": {"field": "current leaderboard score", "operator": ">", "value": 1000},
        "selected_count": len(selected),
        "selected": selected,
    }
    def save() -> None:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        temporary = OUTPUT.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, OUTPUT)

    save()
    for index, row in enumerate(selected):
        error = None
        submissions = None
        for delay in (0, 2, 5, 10, 20, 40):
            if delay:
                time.sleep(delay)
            try:
                submissions = api.competition_team_submissions(row["team_id"]) or []
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if submissions is None:
            row["mapping_status"] = "error"
            row["mapping_error"] = error
        else:
            row["active_submissions"] = [
                {
                    "submission_id": int(field(item, "id")),
                    "date_submitted": str(field(item, "date_submitted", "dateSubmitted") or ""),
                    "public_score": field(item, "public_score", "publicScore"),
                }
                for item in submissions
            ]
            row["mapping_status"] = "ok"
        if index % 10 == 0:
            save()
        time.sleep(1.0)
    save()
    print(json.dumps({"rows": len(entries), "selected": len(selected), "submission_count": sum(len(x["active_submissions"]) for x in selected)}))


if __name__ == "__main__":
    main()
