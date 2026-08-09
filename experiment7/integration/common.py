from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REFERENCE_ARCHIVE_NAME = "experiment7_code_for_gpt_2026-08-08.zip"
REFERENCE_ARCHIVE_BYTES = 94_038
REFERENCE_ARCHIVE_SHA256 = (
    "9c0d24067eacee8abc38223dba28d893e5d1e4e9b75204a9ce92a03093558229"
)
DEFAULT_SOURCE_BRANCH = "agent/experiment7-training-ready-20260809"
DEFAULT_WORK_BRANCH = "codex/experiment7-multideck-run-20260809"


class Experiment7Error(RuntimeError):
    """Raised for a contract violation that must stop an Experiment 7 stage."""


def sha256_file(path: str | Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_runtime_files(root: str | Path) -> list[Path]:
    """Return source/runtime files while ignoring interpreter-generated caches."""
    base = Path(root)
    return [
        path
        for path in sorted(base.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.relative_to(base).parts
        and path.suffix not in {".pyc", ".pyo"}
    ]


def directory_sha256(root: str | Path) -> str:
    base = Path(root)
    digest = hashlib.sha256()
    for path in stable_runtime_files(base):
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def canonical_deck(cards: Sequence[int]) -> tuple[int, ...]:
    values = tuple(sorted(int(card) for card in cards))
    if len(values) != 60:
        raise Experiment7Error(f"deck must contain exactly 60 cards, got {len(values)}")
    if any(card <= 0 for card in values):
        raise Experiment7Error("deck contains a non-positive card ID")
    return values


def canonical_deck_payload(cards: Sequence[int]) -> bytes:
    return json.dumps(
        canonical_deck(cards), separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def canonical_deck_sha256(cards: Sequence[int]) -> str:
    return hashlib.sha256(canonical_deck_payload(cards)).hexdigest()


def write_deck(path: str | Path, cards: Sequence[int]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"{card}\n" for card in canonical_deck(cards)), encoding="utf-8")
    return target


def read_deck(path: str | Path) -> tuple[int, ...]:
    values = [
        int(line.strip())
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    return canonical_deck(values)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding=encoding, dir=target.parent, delete=False, newline=""
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, target)


def write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_version(value: str) -> tuple[int | str, ...]:
    parts: list[int | str] = []
    for token in re.split(r"([0-9]+)", value or ""):
        if not token:
            continue
        parts.append(int(token) if token.isdigit() else token.lower())
    return tuple(parts)


def parse_timestamp(value: Any, fallback: float = 0.0) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip():
        text = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            pass
    return fallback


def stable_episode_key(row: Mapping[str, Any]) -> tuple[float, int, str]:
    try:
        episode = int(row.get("episode_id", 0))
    except (TypeError, ValueError):
        episode = 0
    return (
        parse_timestamp(row.get("create_time")),
        episode,
        str(row.get("raw_path", "")),
    )


def is_forced_decision(options: Sequence[Any], minimum: int, maximum: int) -> bool:
    count = len(options)
    minimum = max(0, min(count, int(minimum)))
    maximum = max(minimum, min(count, int(maximum)))
    if minimum == maximum == 0:
        return True
    if minimum == maximum == count:
        return True
    return count == 1 and minimum == maximum == 1


def validate_action(action: Any, options: Sequence[Any], minimum: int, maximum: int) -> tuple[bool, str]:
    if not isinstance(action, list):
        return False, "action_not_list"
    if any(not isinstance(index, int) or isinstance(index, bool) for index in action):
        return False, "action_index_not_int"
    if len(action) != len(set(action)):
        return False, "duplicate_action_index"
    if not int(minimum) <= len(action) <= int(maximum):
        return False, "action_count_out_of_bounds"
    if any(index < 0 or index >= len(options) for index in action):
        return False, "action_index_out_of_bounds"
    return True, ""


def wilson_interval(wins: int, losses: int, draws: int, z: float = 1.96) -> tuple[float, float]:
    n = int(wins) + int(losses) + int(draws)
    if n <= 0:
        return 0.0, 0.0
    score = (wins + 0.5 * draws) / n
    denominator = 1.0 + z * z / n
    center = (score + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(
        score * (1.0 - score) / n + z * z / (4.0 * n * n)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def run_checked(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    log_path: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    rendered = [os.fspath(value) for value in command]
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(key): str(value) for key, value in env.items()})
    completed = subprocess.run(
        rendered,
        cwd=None if cwd is None else os.fspath(cwd),
        env=merged_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if log_path is not None:
        target = Path(log_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        tail = completed.stdout[-4000:]
        raise Experiment7Error(
            f"command failed ({completed.returncode}): {' '.join(rendered)}\n{tail}"
        )
    return completed


def resolve_python(value: str | Path | None = None) -> str:
    if value:
        return os.fspath(value)
    return os.environ.get("PYTHON", sys.executable)


def find_unique_file(root: str | Path, filename: str) -> Path:
    matches = sorted(Path(root).rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found under {root}")
    if len(matches) > 1:
        exact = [path for path in matches if path.parent == Path(root)]
        if len(exact) == 1:
            return exact[0]
        raise Experiment7Error(
            f"ambiguous {filename} under {root}: {[str(path) for path in matches]}"
        )
    return matches[0]


def counter_dict(values: Iterable[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items(), key=lambda item: str(item[0]))}


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    root: Path
    features: Path
    decisions: Path
    catalog: Path
    token_cache: Path
    sequence_cache: Path
    identity_cache: Path
    calibration_episodes: int
    deck_sha256: str | None = None
    deck_path: Path | None = None

    def vendor_source_args(self) -> list[str]:
        return [
            self.name,
            str(self.features),
            str(self.token_cache),
            str(self.sequence_cache),
            str(self.identity_cache),
            str(self.calibration_episodes),
        ]

    def verify_source_args(self) -> list[str]:
        return [
            self.name,
            str(self.features),
            str(self.token_cache),
            str(self.sequence_cache),
            str(self.identity_cache),
        ]
