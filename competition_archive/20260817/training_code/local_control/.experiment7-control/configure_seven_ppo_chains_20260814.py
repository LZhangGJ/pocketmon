from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-async-ppo-league-20260811")
TRAINING = Path("/dataT0/Free/lzhang/pocketmon-runs/experiment7-universal-incremental-20260812/0812-d14-ram-npz-fast-20260813")
LAUNCHER = Path("/homes/lzhang/worktrees/experiment7-async-4ppo-f8824b0/experiment7/integration")
POOL = ROOT / "control/universal-bc-deck-pool-20260813/universal_bc_decks.json"
CONTROL = ROOT / "control/seven-ppo-distributed-20260814"

sys.path.insert(0, str(LAUNCHER))
from async_ppo_control import add_chain, atomic_write_json, read_json, utc_now  # noqa: E402


def chain_config(
    *,
    deck_name: str,
    archetype_id: str,
    archetype_label: str,
    deck_path: str,
    deck_sha: str,
    checkpoint: str,
    learner_deck_pool: str | None = None,
    long_game: bool = False,
) -> dict:
    rollout = {
        "selfPlayFraction": 0.15,
        "learnerSeat1Fraction": 0.55,
        "archetypeWeights": {"A02": 1.35, "A08": 1.35, "LUCARIO_GOLD": 1.25},
        "agentWeights": {},
        "longGameMinPlayerDecisions": 70 if long_game else 0,
        "longGameWeight": 1.5 if long_game else 1.0,
        "weightPolicy": "distributed all-chain rollout; bounded evidence-driven weighting",
    }
    result = {
        "deckName": deck_name,
        "archetypeId": archetype_id,
        "archetypeLabel": archetype_label,
        "deckPath": deck_path,
        "deckSha256": deck_sha,
        "teacher": checkpoint,
        "current": {"generation": 0, "checkpoint": checkpoint},
        "trainingControl": {
            "schemaVersion": 2,
            "updatedAt": utc_now(),
            "sourceRoundId": "dual-bc-admission-20260814",
            "evidence": {
                "engineSeedControlled": False,
                "reason": "user-authorized dual BC admission and distributed PPO evolution",
            },
            "rollout": rollout,
            "learner": {
                "minDecisions": 12000,
                "maxBehaviorLag": 2,
                "teacherAnchorCoefficient": 0.04,
                "seat1Weight": 1.25,
                "learningRate": 5e-6,
                "ppoEpochs": 1,
                "normalizeAdvantagesByPlayer": True,
                "balancePlayerMinibatches": True,
            },
        },
        "bootstrap": {
            "reason": "latest admitted Universal BC frozen twin to continuously evolving PPO",
            "hashVerificationSkipped": True,
        },
    }
    if learner_deck_pool:
        result["learnerDeckPool"] = learner_deck_pool
    return result


def main() -> None:
    CONTROL.mkdir(parents=True, exist_ok=True)
    league_path = ROOT / "state/league.json"
    manifest = read_json(POOL)
    by_sha = {row["deckSha256"]: row for row in manifest["selected"]}
    gold_sha = "dc8571d0bc2e546a1f85b938696cfc40a1451c68a4ccc1f695e7c3e1c74f1278"
    std_sha = "606a775392ffe25e058b19c17801d58a4bf30f7cd8c62782388d3de7e7eb5283"
    large_sha = "89e6155f25310ee695c0761c85d3ae8e44f376456ff0539231820f8e803f2d5e"
    specs = {
        "universal_ppo_standard_1m": chain_config(
            deck_name="universal_standard_alakazam_representative",
            archetype_id="UNIVERSAL_STANDARD",
            archetype_label="Universal standard 1.05M / 133-deck evolving PPO",
            deck_path=by_sha[std_sha]["deckPath"],
            deck_sha=std_sha,
            checkpoint=str(TRAINING / "standard_1m/best_model.pt"),
            learner_deck_pool=str(POOL),
        ),
        "universal_ppo_large_256x6": chain_config(
            deck_name="universal_large_dragapult_representative",
            archetype_id="UNIVERSAL_LARGE",
            archetype_label="Universal large 6.44M / 133-deck evolving PPO",
            deck_path=by_sha[large_sha]["deckPath"],
            deck_sha=large_sha,
            checkpoint=str(TRAINING / "large_256x6/best_model.pt"),
            learner_deck_pool=str(POOL),
        ),
        "lucario_gold_exact": chain_config(
            deck_name="lucario_gold_exact",
            archetype_id="LUCARIO_GOLD",
            archetype_label="Mega Lucario / Hariyama / Solrock-Lunatone gold exact",
            deck_path=by_sha[gold_sha]["deckPath"],
            deck_sha=gold_sha,
            checkpoint=str(TRAINING / "large_256x6/best_model.pt"),
            long_game=True,
        ),
    }
    before = read_json(league_path)
    added = []
    for name, config in specs.items():
        config_path = CONTROL / f"{name}.json"
        atomic_write_json(config_path, config)
        league = read_json(league_path)
        if name not in league["chains"]:
            add_chain(league_path, name, config_path)
            added.append(name)
    after = read_json(league_path)
    receipt = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "status": "configured",
        "beforeChains": list(before["chains"]),
        "addedChains": added,
        "activeChains": list(after["chains"]),
        "distributedRollout": {
            "allChains": True,
            "cpuLimitPercent": 95,
            "ioPressureLimitPercent": 80,
            "loadGuardRequired": True,
        },
        "oldLucarioGold": {
            "preserved": True,
            "latestGeneration": 113,
            "role": "historical anchor only",
        },
    }
    atomic_write_json(CONTROL / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
