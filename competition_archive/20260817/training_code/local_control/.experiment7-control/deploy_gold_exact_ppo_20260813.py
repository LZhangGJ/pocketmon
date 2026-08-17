#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


CHAIN = "lucario_gold_exact"
PARENT_CHAIN = "mega_lucario_ex"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league-root", type=Path, required=True)
    parser.add_argument("--integration", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(args.integration.resolve()))
    from async_ppo_control import add_chain, atomic_write_json, read_json  # noqa: PLC0415

    league_root = args.league_root.resolve()
    league_path = league_root / "state/league.json"
    league = read_json(league_path)
    deck = args.deck.resolve()
    cards = [int(line) for line in deck.read_text().splitlines() if line.strip()]
    if len(cards) != 60 or any(card <= 0 for card in cards):
        raise ValueError(f"gold_exact deck must contain 60 positive card IDs: {deck}")

    if CHAIN in league["chains"]:
        print(json.dumps({"status": "already_installed", "chain": CHAIN}))
        return
    parent = league["chains"][PARENT_CHAIN]
    config = {
        "deckName": CHAIN,
        "archetypeId": "LUCARIO_GOLD",
        "archetypeLabel": "Mega Lucario ex / gold_exact optimized deck",
        "deckPath": str(deck),
        # Required by the league schema as an identity key; it is not used as
        # an experimental admission/blocking gate.
        "deckSha256": hashlib.sha256(
            json.dumps(sorted(cards), separators=(",", ":"), ensure_ascii=True).encode("ascii")
        ).hexdigest(),
        "teacher": parent["teacher"],
        "current": {
            "generation": 0,
            "checkpoint": parent["current"]["checkpoint"],
        },
        "trainingControl": {
            "schemaVersion": 1,
            "updatedAt": now(),
            "sourceRoundId": "bootstrap-gold-exact-20260813",
            "evidence": {
                "generation": 0,
                "progress": "bootstrap",
                "engineSeedControlled": False,
            },
            "rollout": {
                "selfPlayFraction": 0.25,
                "learnerSeat1Fraction": 0.50,
                "archetypeWeights": {},
                "agentWeights": {},
                "weightPolicy": "bootstrap uniform; adaptive after first complete matrix",
            },
            "learner": {
                "minDecisions": 8000,
                "maxBehaviorLag": 2,
                "teacherAnchorCoefficient": 0.04,
                "seat1Weight": 1.0,
                "learningRate": 1e-5,
                "ppoEpochs": 1,
                "normalizeAdvantagesByPlayer": True,
                "balancePlayerMinibatches": True,
            },
        },
        "bootstrap": {
            "parentChain": PARENT_CHAIN,
            "parentSnapshotId": parent["current"]["snapshotId"],
            "reason": "same model architecture; new fixed competition deck",
            "hashVerificationSkipped": True,
        },
    }
    config_path = league_root / "state/lucario-gold-exact-chain.json"
    atomic_write_json(config_path, config)
    installed = add_chain(league_path, CHAIN, config_path)
    receipt = league_root / "monitoring/lucario-gold-exact/installed.json"
    atomic_write_json(
        receipt,
        {
            "schemaVersion": 1,
            "status": "installed_waiting_for_bootstrap_deployment",
            "installedAt": now(),
            "chain": CHAIN,
            "deck": str(deck),
            "cards": len(cards),
            "parent": installed["bootstrap"],
        },
    )
    print(json.dumps({"status": "installed", "chain": CHAIN, "receipt": str(receipt)}))


if __name__ == "__main__":
    main()
