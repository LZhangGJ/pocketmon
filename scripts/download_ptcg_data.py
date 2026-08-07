from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

INDEX_SLUG = "kaggle/pokemon-tcg-ai-battle-episodes-index"
DAILY_SLUG_TEMPLATE = "kaggle/pokemon-tcg-ai-battle-episodes-{date}"
MANIFEST = "manifest.csv"

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "replays"


def run_kaggle(*args: str) -> None:
    cmd = [sys.executable, "-m", "kaggle", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        if "credentials" in message.lower() or "unauthorized" in message.lower():
            raise RuntimeError(
                "Kaggle auth failed. Set KAGGLE_USERNAME with KAGGLE_KEY "
                "(or KAGGLE_API_TOKEN) in .env, or place kaggle.json in ~/.kaggle/."
            )
        raise RuntimeError(message)


def download_file(
    slug: str,
    file_name: str,
    target_dir: Path,
    force: bool = False,
    max_retries: int = 3,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / file_name
    if target_path.exists() and not force:
        return target_path

    if target_path.exists() and force:
        target_path.unlink()

    last_error: Exception | None = None
    for _ in range(max_retries):
        try:
            run_kaggle("datasets", "download", slug, "-f", file_name, "-p", str(target_dir))

            archive = target_dir / f"{file_name}.zip"
            if archive.exists():
                with zipfile.ZipFile(archive, "r") as zf:
                    zf.extractall(target_dir)
                archive.unlink()

            if not target_path.exists():
                raise FileNotFoundError(f"{file_name} was not found after download from {slug}")
            return target_path
        except Exception as exc:
            last_error = exc
            archive = target_dir / f"{file_name}.zip"
            if archive.exists():
                archive.unlink()
            if target_path.exists() and force:
                target_path.unlink()
            if "404" in str(exc):
                break
            continue

    raise RuntimeError(f"Failed to download {file_name} from {slug} after {max_retries} retries: {last_error}")


def download_dataset(slug: str, target_dir: Path) -> Path:
    """Download and unzip a full daily dataset with one Kaggle API request."""
    target_dir.mkdir(parents=True, exist_ok=True)
    run_kaggle("datasets", "download", slug, "-p", str(target_dir), "--unzip")
    manifest = target_dir / MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(f"{MANIFEST} was not found after bulk download from {slug}")
    return manifest


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_daily_slug(requested_date: str | None, index_rows: list[dict[str, str]]) -> tuple[str, str]:
    if not index_rows:
        raise RuntimeError("Index manifest is empty")

    if requested_date:
        selected = next((row for row in index_rows if row.get("date") == requested_date), None)
        if selected is None:
            raise RuntimeError(f"date={requested_date} not found in index manifest")
    else:
        selected = index_rows[-1]

    date = selected["date"]
    slug_value = selected.get("daily_dataset_slug") or DAILY_SLUG_TEMPLATE.format(date=date)
    slug = slug_value if "/" in slug_value else f"kaggle/{slug_value}"
    return date, slug


def select_episodes(episodes: list[dict[str, str]], max_episodes: int) -> list[dict[str, str]]:
    if max_episodes < 0:
        raise ValueError("max episodes must be non-negative")
    return episodes if max_episodes == 0 else episodes[:max_episodes]


def main() -> None:
    load_dotenv(ROOT / ".env")
    if not os.getenv("KAGGLE_KEY") and os.getenv("KAGGLE_API_TOKEN"):
        os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_TOKEN"]

    parser = argparse.ArgumentParser(description="Download PTCG AI Battle daily datasets from Kaggle")
    parser.add_argument("--date", default=None, help="Date in YYYY-MM-DD. Default: latest available")
    parser.add_argument("--max-episodes", type=int, default=50, help="How many top episodes to download; 0 means all")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="Replay root directory")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Finish the batch and audit files listed by the manifest but unavailable from Kaggle",
    )
    parser.add_argument("--audit-output", type=Path, default=None, help="Optional JSON audit path")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    index_dir = data_dir / "_index"
    index_manifest = download_file(INDEX_SLUG, MANIFEST, index_dir, force=True)
    index_rows = read_manifest(index_manifest)

    date, daily_slug = resolve_daily_slug(args.date, index_rows)
    day_dir = data_dir / date

    daily_manifest = (
        download_dataset(daily_slug, day_dir)
        if args.max_episodes == 0
        else download_file(daily_slug, MANIFEST, day_dir, force=True)
    )
    episodes = read_manifest(daily_manifest)

    score_key = next((k for k in ("avg_score", "score", "top_avg_score") if episodes and k in episodes[0]), None)
    if score_key:
        episodes.sort(key=lambda row: -float(row.get(score_key) or 0))

    id_key = next((k for k in ("episode_id", "EpisodeId", "id") if episodes and k in episodes[0]), None)
    if id_key is None:
        raise RuntimeError("Cannot detect episode id column from daily manifest")

    selected = select_episodes(episodes, args.max_episodes)
    available = 0
    failures: list[dict[str, str]] = []
    for row in selected:
        episode_id = row[id_key]
        try:
            download_file(daily_slug, f"{episode_id}.json", day_dir)
            available += 1
        except RuntimeError as exc:
            failures.append({"episode_id": str(episode_id), "error": str(exc)})
            if not args.allow_missing:
                raise

    audit = {
        "schema_version": 1,
        "date": date,
        "daily_slug": daily_slug,
        "manifest_episodes": len(episodes),
        "selected_episodes": len(selected),
        "available_episodes": available,
        "missing_episodes": len(failures),
        "missing": failures,
        "complete": not failures and available == len(selected),
        "output_dir": str(day_dir),
    }
    audit_output = (args.audit_output or (data_dir / "_audit" / f"{date}.json")).expanduser().resolve()
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"date={date}")
    print(f"daily_slug={daily_slug}")
    print(f"episodes_available={available}")
    print(f"episodes_missing={len(failures)}")
    print(f"output_dir={day_dir}")
    print(f"audit_output={audit_output}")


if __name__ == "__main__":
    main()
