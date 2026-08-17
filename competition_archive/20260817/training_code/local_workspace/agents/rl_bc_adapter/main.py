from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.agent_adapter import RLBCPolicyAdapter


def _load_rule_agent():
    path = ROOT / "agents" / "lucario_rule" / "main.py"
    spec = importlib.util.spec_from_file_location("rl_bc_lucario_fallback", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load rule fallback from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RULE_AGENT = _load_rule_agent()
CHECKPOINT = Path(os.environ.get(
    "POCKETMON_RL_CHECKPOINT",
    ROOT / "checkpoints" / "rl_bc_002" / "02782c3" / "a" / "seed_20260720_best.pt",
))
POLICY = RLBCPolicyAdapter(
    CHECKPOINT,
    fallback=RULE_AGENT.agent,
    device=os.environ.get("POCKETMON_RL_DEVICE", "cpu"),
)


def diagnostics() -> dict:
    return POLICY.diagnostics()


def agent(observation: dict) -> list[int]:
    return POLICY.act(observation)
