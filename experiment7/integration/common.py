from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_deck_signature(cards: Sequence[int]) -> tuple[tuple[int, int], ...]:
    values = [int(value) for value in cards]
    if len(values) != 60 or any(value <= 0 for value in values):
        raise ValueError("a deck must contain exactly 60 positive card IDs")
    return tuple(sorted(Counter(values).items()))


def canonical_deck_cards(cards: Sequence[int]) -> list[int]:
    signature = canonical_deck_signature(cards)
    return [card_id for card_id, count in signature for _ in range(count)]


def canonical_deck_sha256(cards: Sequence[int]) -> str:
    signature = canonical_deck_signature(cards)
    payload = json.dumps(signature, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def signature_id(cards: Sequence[int], prefix: str = "deck") -> str:
    signature = canonical_deck_signature(cards)
    payload = json.dumps(signature, ensure_ascii=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def write_deck(path: Path, cards: Sequence[int]) -> None:
    values = canonical_deck_cards(cards)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(value) for value in values) + "\n", encoding="utf-8")


def read_deck(path: Path) -> list[int]:
    values = [int(line.strip()) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    canonical_deck_signature(values)
    return values


def open_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_slug(value: str, fallback: str = "item") -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "-", value).strip("-").lower()
    return slug[:80] or fallback


def first_present(mapping: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
