from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


_SOURCE_FILE = globals().get("__file__")
if _SOURCE_FILE:
    PACKAGE_ROOT = Path(_SOURCE_FILE).resolve().parent
elif Path("/kaggle_simulations/agent").is_dir():
    # Kaggle's simulation runner executes main.py with exec(), so __file__ is
    # intentionally absent even though the extracted package has a fixed root.
    PACKAGE_ROOT = Path("/kaggle_simulations/agent")
else:
    # This also makes a faithful no-__file__ preflight possible outside Kaggle.
    PACKAGE_ROOT = Path.cwd().resolve()
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from rl.agent_adapter import RLBCPolicyAdapter


def _read_deck(path: Path) -> list[int]:
    deck = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck must contain exactly 60 cards, got {len(deck)}")
    return deck


def _read_q_policy(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    import json

    policy = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(policy, dict):
        raise ValueError("q_policy.json must contain an object")
    return policy


DECK = _read_deck(PACKAGE_ROOT / "deck.csv")
CHECKPOINT = Path(os.environ.get("POCKETMON_RL_CHECKPOINT", PACKAGE_ROOT / "checkpoint.pt"))
Q_CHECKPOINT = PACKAGE_ROOT / "action_q.pt"
Q_POLICY = _read_q_policy(PACKAGE_ROOT / "q_policy.json")


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
    q_checkpoint_path=Q_CHECKPOINT if Q_CHECKPOINT.is_file() else None,
    q_top_k=int(os.environ.get("POCKETMON_Q_TOP_K", Q_POLICY.get("top_k", 4))),
    q_uncertainty_penalty=float(os.environ.get(
        "POCKETMON_Q_UNCERTAINTY_PENALTY", Q_POLICY.get("uncertainty_penalty", 0.25)
    )),
    q_min_margin=float(os.environ.get(
        "POCKETMON_Q_MIN_MARGIN", Q_POLICY.get("min_margin", 0.15)
    )),
    q_max_uncertainty=float(os.environ.get(
        "POCKETMON_Q_MAX_UNCERTAINTY", Q_POLICY.get("max_uncertainty", 0.15)
    )),
    q_max_override_rate=float(os.environ.get(
        "POCKETMON_Q_MAX_OVERRIDE_RATE", Q_POLICY.get("max_override_rate", 1.0)
    )),
    q_min_validation_rows=int(os.environ.get(
        "POCKETMON_Q_MIN_VALIDATION_ROWS", Q_POLICY.get("min_validation_rows", 0)
    )),
    q_max_validation_mae=float(os.environ.get(
        "POCKETMON_Q_MAX_VALIDATION_MAE", Q_POLICY.get("max_validation_mae", "inf")
    )),
)


def diagnostics() -> dict[str, Any]:
    result = POLICY.diagnostics()
    result["package_root"] = str(PACKAGE_ROOT)
    return result


# Kaggle Environments selects the last callable left by exec(main.py), not a
# function named "agent". Keep the submission entry point last in this file.
def agent(observation: dict[str, Any]) -> list[int]:
    return POLICY.act(observation)
