from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from async_ppo_control import atomic_write_json, read_json, utc_now


def resource_snapshot() -> tuple[float, float]:
    cores = os.cpu_count() or 1
    load_one = os.getloadavg()[0]
    cpu = 100.0 * load_one / cores
    io = 0.0
    pressure = Path("/proc/pressure/io")
    if pressure.is_file():
        for token in pressure.read_text(encoding="utf-8").splitlines()[0].split():
            if token.startswith("avg10="):
                io = float(token.split("=", 1)[1])
                break
    return cpu, io


def wait_for_resources(cpu_limit: float, io_limit: float, worker_id: str) -> None:
    while True:
        cpu, io = resource_snapshot()
        status = "PASS" if cpu < cpu_limit and io < io_limit else "WAIT"
        print(
            json.dumps(
                {
                    "event": f"RESOURCE_GUARD_{status}",
                    "worker": worker_id,
                    "cpu": round(cpu, 2),
                    "io": round(io, 2),
                    "cpuLimit": cpu_limit,
                    "ioLimit": io_limit,
                    "at": utc_now(),
                }
            ),
            flush=True,
        )
        if status == "PASS":
            return
        time.sleep(30)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously collect asynchronous PPO shards on CPU")
    parser.add_argument("--league", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--buffer-root", type=Path, required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--worker-id", default=socket.gethostname())
    parser.add_argument("--episodes-per-shard", type=int, default=20)
    parser.add_argument("--refresh-rounds", type=int, default=1)
    parser.add_argument("--self-play-fraction", type=float, default=0.25)
    parser.add_argument("--cpu-limit", type=float, default=70.0)
    parser.add_argument("--io-limit", type=float, default=80.0)
    parser.add_argument(
        "--only-chain",
        action="append",
        default=[],
        help="Restrict this worker to one or more chain names; repeat as needed",
    )
    args = parser.parse_args()
    if args.episodes_per_shard <= 0 or args.refresh_rounds <= 0:
        raise ValueError("shard and refresh sizes must be positive")
    collector = args.worktree / "experiment7/integration/collect_universal_ppo_rollouts.py"
    sequence = 0
    env = dict(os.environ)
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    # Some doraemon nodes expose an older system libstdc++ even though the
    # official engine was built with GLIBCXX_3.4.29.  Keep every collector on
    # the already-installed conda runtime ABI instead of letting host order
    # decide whether libcg.so can be loaded.
    compat_lib = Path(
        "/homes/lzhang/mypath/new/pkgs/libstdcxx-ng-11.2.0-h1234567_1/lib"
    )
    if (compat_lib / "libstdc++.so.6").is_file():
        inherited = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{compat_lib}:{inherited}" if inherited else str(compat_lib)
        )
    while True:
        league = read_json(args.league)
        requested_chains = set(args.only_chain)
        missing_chains = requested_chains.difference(league["chains"])
        if missing_chains:
            raise ValueError(f"unknown --only-chain values: {sorted(missing_chains)}")
        pool = Path(league["poolPath"])
        for _ in range(args.refresh_rounds):
            for chain_name, chain in sorted(league["chains"].items()):
                if requested_chains and chain_name not in requested_chains:
                    continue
                current = chain["current"]
                rollout_control = chain.get("trainingControl", {}).get("rollout", {})
                if not bool(rollout_control.get("enabled", True)):
                    print(
                        json.dumps(
                            {
                                "event": "ROLLOUT_CHAIN_DISABLED",
                                "worker": args.worker_id,
                                "chain": chain_name,
                                "at": utc_now(),
                            }
                        ),
                        flush=True,
                    )
                    continue
                wait_for_resources(args.cpu_limit, args.io_limit, args.worker_id)
                self_play_fraction = float(
                    rollout_control.get("selfPlayFraction", args.self_play_fraction)
                )
                learner_seat1_fraction = float(
                    rollout_control.get("learnerSeat1Fraction", 0.5)
                )
                tactical_shaping_profile = str(
                    rollout_control.get("tacticalShapingProfile", "none")
                )
                tactical_shaping_revision = int(
                    rollout_control.get("tacticalShapingRevision", 0)
                )
                boss_reservation_penalty = float(
                    rollout_control.get("bossReservationPenalty", 0.0)
                )
                boss_reservation_preference = bool(
                    rollout_control.get("bossReservationPreference", False)
                )
                boss_post_play_penalty = float(
                    rollout_control.get("bossPostPlayPenalty", 0.0)
                )
                boss_post_play_preference = bool(
                    rollout_control.get("bossPostPlayPreference", False)
                )
                a02_poffin_decline_penalty = float(
                    rollout_control.get("a02PoffinDeclinePenalty", 0.0)
                )
                a02_poffin_preference = bool(
                    rollout_control.get("a02PoffinPreference", False)
                )
                a02_munkidori_overfill_penalty = float(
                    rollout_control.get("a02MunkidoriOverfillPenalty", 0.05)
                )
                a02_bench_budget_preference = bool(
                    rollout_control.get("a02BenchBudgetPreference", False)
                )
                a02_outcome_gated_ordering = bool(
                    rollout_control.get("a02OutcomeGatedOrdering", False)
                )
                a02_projected_bench_budget = bool(
                    rollout_control.get("a02ProjectedBenchBudget", False)
                )
                successor_attach_preference = bool(
                    rollout_control.get("successorAttachPreference", False)
                )
                a08_terminal_before_evolve_mode = str(
                    rollout_control.get("a08TerminalBeforeEvolveMode", "control")
                )
                a08_gated_attack_penalty = float(
                    rollout_control.get("a08GatedAttackPenalty", 0.10)
                )
                a08_maximum_belt_support_penalty = float(
                    rollout_control.get("a08MaximumBeltSupportPenalty", 0.0)
                )
                a08_maximum_belt_preference = bool(
                    rollout_control.get("a08MaximumBeltPreference", False)
                )
                a08_second_attacker_reward = float(
                    rollout_control.get("a08SecondAttackerReward", 0.0)
                )
                a08_recovery_end_penalty = float(
                    rollout_control.get("a08RecoveryEndPenalty", 0.0)
                )
                a08_recovery_preference = bool(
                    rollout_control.get("a08RecoveryPreference", False)
                )
                end_with_attack_penalty = float(
                    rollout_control.get("endWithAttackPenalty", 0.0)
                )
                end_with_attack_preference = bool(
                    rollout_control.get("endWithAttackPreference", False)
                )
                lucario_evolve_penalty = float(
                    rollout_control.get("lucarioEvolvePenalty", 0.0)
                )
                lucario_attach_penalty = float(
                    rollout_control.get("lucarioAttachPenalty", 0.0)
                )
                lucario_aura_overkill_penalty = float(
                    rollout_control.get("lucarioAuraOverkillPenalty", 0.0)
                )
                lucario_aura_hard_mask = bool(
                    rollout_control.get("lucarioAuraHardMask", True)
                )
                lucario_ordering_preference = bool(
                    rollout_control.get("lucarioOrderingPreference", False)
                )
                dragapult_ready_attacker_penalty = float(
                    rollout_control.get("dragapultReadyAttackerPenalty", 0.14)
                )
                dragapult_evolve_penalty = float(
                    rollout_control.get("dragapultEvolvePenalty", 0.16)
                )
                dragapult_wall_penalty = float(
                    rollout_control.get("dragapultWallPenalty", 0.10)
                )
                dragapult_budew_overstay_penalty = float(
                    rollout_control.get("dragapultBudewOverstayPenalty", 0.08)
                )
                dragapult_resource_penalty = float(
                    rollout_control.get("dragapultResourcePenalty", 0.08)
                )
                dragapult_wall_preference = bool(
                    rollout_control.get("dragapultWallPreference", True)
                )
                dragapult_terminal_search_depth = int(
                    rollout_control.get("dragapultTerminalSearchDepth", 2)
                )
                dragapult_search_bias_scale = float(
                    rollout_control.get("dragapultSearchBiasScale", 1.0)
                )
                long_game_min_player_decisions = int(
                    rollout_control.get("longGameMinPlayerDecisions", 0)
                )
                long_game_weight = float(rollout_control.get("longGameWeight", 1.0))
                sequence += 1
                stamp = time.time_ns()
                shard_id = f"{args.worker_id}-{sequence:08d}-{stamp}"
                output = args.buffer_root / "ready" / chain_name / f"{shard_id}.jsonl.gz"
                output.parent.mkdir(parents=True, exist_ok=True)
                log = args.buffer_root / "logs" / args.worker_id / f"{shard_id}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                seed = (stamp ^ hash((args.worker_id, chain_name, sequence))) & 0x7FFFFFFF
                command = [
                    args.python,
                    str(collector),
                    "--reference-root",
                    str(Path(league["referenceRoot"])),
                    "--engine-catalog",
                    str(Path(league["engineCatalog"])),
                    "--checkpoint",
                    str(Path(current["checkpoint"])),
                    "--teacher",
                    str(Path(chain["teacher"])),
                    "--deck",
                    str(Path(chain["deckPath"])),
                    "--pool",
                    str(pool),
                    "--cg-dir",
                    str(Path(league["cgDir"])),
                    "--episodes",
                    str(args.episodes_per_shard),
                    "--self-play-fraction",
                    str(self_play_fraction),
                    "--learner-seat1-fraction",
                    str(learner_seat1_fraction),
                    "--archetype-weights-json",
                    json.dumps(rollout_control.get("archetypeWeights", {}), separators=(",", ":")),
                    "--agent-weights-json",
                    json.dumps(rollout_control.get("agentWeights", {}), separators=(",", ":")),
                    "--tactical-shaping-profile",
                    tactical_shaping_profile,
                    "--tactical-shaping-revision",
                    str(tactical_shaping_revision),
                    "--boss-reservation-penalty",
                    str(boss_reservation_penalty),
                    "--boss-post-play-penalty",
                    str(boss_post_play_penalty),
                    "--a02-poffin-decline-penalty",
                    str(a02_poffin_decline_penalty),
                    "--a02-munkidori-overfill-penalty",
                    str(a02_munkidori_overfill_penalty),
                    "--a08-terminal-before-evolve-mode",
                    a08_terminal_before_evolve_mode,
                    "--a08-gated-attack-penalty",
                    str(a08_gated_attack_penalty),
                    "--a08-maximum-belt-support-penalty",
                    str(a08_maximum_belt_support_penalty),
                    "--a08-second-attacker-reward",
                    str(a08_second_attacker_reward),
                    "--a08-recovery-end-penalty",
                    str(a08_recovery_end_penalty),
                    "--end-with-attack-penalty",
                    str(end_with_attack_penalty),
                    "--lucario-evolve-penalty",
                    str(lucario_evolve_penalty),
                    "--lucario-attach-penalty",
                    str(lucario_attach_penalty),
                    "--lucario-aura-overkill-penalty",
                    str(lucario_aura_overkill_penalty),
                    "--dragapult-ready-attacker-penalty",
                    str(dragapult_ready_attacker_penalty),
                    "--dragapult-evolve-penalty",
                    str(dragapult_evolve_penalty),
                    "--dragapult-wall-penalty",
                    str(dragapult_wall_penalty),
                    "--dragapult-budew-overstay-penalty",
                    str(dragapult_budew_overstay_penalty),
                    "--dragapult-resource-penalty",
                    str(dragapult_resource_penalty),
                    "--dragapult-terminal-search-depth",
                    str(dragapult_terminal_search_depth),
                    "--dragapult-search-bias-scale",
                    str(dragapult_search_bias_scale),
                    "--long-game-min-player-decisions",
                    str(long_game_min_player_decisions),
                    "--long-game-weight",
                    str(long_game_weight),
                    "--temperature",
                    "1.0",
                    "--max-decisions",
                    "5000",
                    "--seed",
                    str(seed),
                    "--run-id",
                    shard_id,
                    "--behavior-generation",
                    str(current["generation"]),
                    "--behavior-snapshot-id",
                    str(current["snapshotId"]),
                    "--role",
                    "diversity",
                    "--device",
                    "cpu",
                    "--output",
                    str(output),
                ]
                learner_deck_pool = current.get("deckCohortReceipt") or chain.get(
                    "learnerDeckPool"
                )
                if learner_deck_pool:
                    command.extend(["--learner-deck-pool", str(Path(learner_deck_pool))])
                if boss_reservation_preference:
                    command.append("--boss-reservation-preference")
                if boss_post_play_preference:
                    command.append("--boss-post-play-preference")
                if a02_poffin_preference:
                    command.append("--a02-poffin-preference")
                if a02_bench_budget_preference:
                    command.append("--a02-bench-budget-preference")
                if a02_outcome_gated_ordering:
                    command.append("--a02-outcome-gated-ordering")
                if a02_projected_bench_budget:
                    command.append("--a02-projected-bench-budget")
                if successor_attach_preference:
                    command.append("--successor-attach-preference")
                if a08_maximum_belt_preference:
                    command.append("--a08-maximum-belt-preference")
                if a08_recovery_preference:
                    command.append("--a08-recovery-preference")
                if end_with_attack_preference:
                    command.append("--end-with-attack-preference")
                if lucario_ordering_preference:
                    command.append("--lucario-ordering-preference")
                command.append(
                    "--lucario-aura-hard-mask"
                    if lucario_aura_hard_mask
                    else "--no-lucario-aura-hard-mask"
                )
                command.append(
                    "--dragapult-wall-preference"
                    if dragapult_wall_preference
                    else "--no-dragapult-wall-preference"
                )
                started = time.perf_counter()
                with log.open("w", encoding="utf-8") as handle:
                    completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
                if completed.returncode:
                    atomic_write_json(
                        output.with_suffix(output.suffix + ".failed.json"),
                        {
                            "createdAt": utc_now(),
                            "worker": args.worker_id,
                            "chain": chain_name,
                            "snapshotId": current["snapshotId"],
                            "returnCode": completed.returncode,
                            "log": str(log.resolve()),
                            "seconds": time.perf_counter() - started,
                        },
                    )
                    continue
        # Refresh every policy checkpoint and the dynamic pool together.


if __name__ == "__main__":
    main()
