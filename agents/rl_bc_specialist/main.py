from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rl.agent_adapter import RLBCPolicyAdapter


def _read_deck(path: Path) -> list[int]:
    deck = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck must contain exactly 60 cards, got {len(deck)}")
    return deck


DECK = _read_deck(PACKAGE_ROOT / "deck.csv")
CHECKPOINT = Path(os.environ.get("POCKETMON_RL_CHECKPOINT", PACKAGE_ROOT / "checkpoint.pt"))


def _legal_fallback(observation: dict[str, Any]) -> list[int]:
    select = observation.get("select")
    if not isinstance(select, dict):
        return list(DECK)
    options = select.get("option")
    if not isinstance(options, list):
        return list(DECK)
    min_count = max(0, int(select.get("minCount", 0)))
    return list(range(min(min_count, len(options))))


POLICY = RLBCPolicyAdapter(
    CHECKPOINT,
    fallback=_legal_fallback,
    device=os.environ.get("POCKETMON_RL_DEVICE", "cpu"),
    confidence_threshold=float(os.environ.get("POCKETMON_RL_CONFIDENCE", "0.0")),
    deck=DECK,
)


def agent(observation: dict[str, Any]) -> list[int]:
    return POLICY.act(observation)


def diagnostics() -> dict[str, Any]:
    result = POLICY.diagnostics()
    result["package_root"] = str(PACKAGE_ROOT)
    return result
