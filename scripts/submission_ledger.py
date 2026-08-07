from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def record_submission(ledger_path: Path, entry: dict[str, Any]) -> dict[str, Any]:
    archive = Path(entry["archive"]).resolve(strict=True)
    actual_sha = sha256(archive)
    supplied_sha = entry.get("archive_sha256")
    if supplied_sha and supplied_sha != actual_sha:
        raise ValueError(f"archive SHA-256 mismatch: expected {supplied_sha}, got {actual_sha}")
    entry = dict(entry)
    entry["archive"] = str(archive)
    entry["archive_sha256"] = actual_sha
    entry["ledger_key"] = f"{entry['date_jst']}:{actual_sha}"
    entry["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()

    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    else:
        ledger = {"schema_version": 1, "competition": "pokemon-tcg-ai-battle", "submissions": []}
    submissions = ledger.setdefault("submissions", [])
    matches = [index for index, row in enumerate(submissions) if row.get("ledger_key") == entry["ledger_key"]]
    if len(matches) > 1:
        raise RuntimeError(f"ledger already contains duplicate key {entry['ledger_key']}")
    if matches:
        prior = submissions[matches[0]]
        prior.update({key: value for key, value in entry.items() if value is not None})
        submissions[matches[0]] = prior
        stored = prior
    else:
        submissions.append(entry)
        stored = entry
    ledger["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(ledger_path, ledger)
    return stored


def main() -> None:
    parser = argparse.ArgumentParser(description="Record or update a hash-deduplicated Kaggle submission ledger")
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--date-jst", required=True)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--agent-package", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--submission-ref", required=True)
    parser.add_argument("--submitted-at-utc", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--public-score", type=float)
    parser.add_argument("--preflight-passed", action="store_true")
    parser.add_argument("--preflight-validator")
    parser.add_argument("--gate-passed", action="store_true")
    parser.add_argument("--gate-report", type=Path)
    args = parser.parse_args()
    entry = {
        "date_jst": args.date_jst,
        "agent_name": args.agent_name,
        "agent_package": str(args.agent_package.resolve()),
        "archive": str(args.archive.resolve()),
        "archive_sha256": args.archive_sha256,
        "submission_ref": str(args.submission_ref),
        "submitted_at_utc": args.submitted_at_utc,
        "description": args.description,
        "status": args.status,
        "public_score": args.public_score,
        "preflight_passed": args.preflight_passed,
        "preflight_validator": args.preflight_validator,
        "gate_passed": args.gate_passed,
        "gate_report": str(args.gate_report.resolve()) if args.gate_report else None,
    }
    print(json.dumps(record_submission(args.ledger.resolve(), entry), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
