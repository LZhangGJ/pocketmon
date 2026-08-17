from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from async_ppo_control import (
    atomic_write_json,
    publish_snapshot,
    read_json,
    sha256_file,
    utc_now,
)
from universal_deck_cohort import materialize_cohort


def read_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schemaVersion": 1, "consumed": {}, "rejected": {}}
    return read_json(path)


def eligible_shards(
    league: dict[str, Any],
    chain_name: str,
    buffer_root: Path,
    ledger: dict[str, Any],
    max_lag: int,
) -> tuple[list[Path], list[dict[str, Any]]]:
    chain = league["chains"][chain_name]
    current_generation = int(chain["current"]["generation"])
    history = [*chain.get("history", []), chain["current"]]
    allowed = {str(row["sha256"]): int(row["generation"]) for row in history}
    ignored = set(ledger.get("consumed", {})) | set(ledger.get("rejected", {}))
    candidates = []
    for summary_path in (buffer_root / "ready" / chain_name).glob("*.jsonl.gz.summary.json"):
        summary = read_json(summary_path)
        rollout_path = Path(summary["output"]["path"])
        key = str(rollout_path.resolve())
        if key in ignored or not rollout_path.is_file():
            continue
        control_enabled = bool(chain.get("trainingControl"))
        is_a08_targeted = (
            chain_name == "a08_dipplin_seaking"
            and rollout_path.name.startswith("a08-targeted-")
        )
        if control_enabled and not summary.get("samplingControl") and not is_a08_targeted:
            ledger.setdefault("rejected", {})[key] = {
                "reason": "missing_dynamic_sampling_control",
                "recordedAt": utc_now(),
            }
            continue
        behavior_sha = str(summary["behaviorCheckpoint"]["sha256"])
        behavior_generation = int(summary.get("behaviorGeneration", -1))
        lag = current_generation - behavior_generation
        if behavior_sha not in allowed or allowed[behavior_sha] != behavior_generation:
            continue
        if lag < 0 or lag > max_lag:
            ledger.setdefault("rejected", {})[key] = {
                "reason": "behavior_lag",
                "lag": lag,
                "recordedAt": utc_now(),
            }
            continue
        # Experiment policy intentionally omits content-hash validation.  The
        # immutable path, snapshot id, behavior generation, and bounded lag
        # remain the acceptance boundary for asynchronous shards.
        candidates.append((behavior_generation, summary_path.stat().st_mtime_ns, rollout_path, summary))
    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [row[2] for row in candidates], [row[3] for row in candidates]


def select_minimum_batch(
    shards: list[Path], summaries: list[dict[str, Any]], min_decisions: int
) -> tuple[list[Path], list[dict[str, Any]], int]:
    selected_paths: list[Path] = []
    selected_summaries: list[dict[str, Any]] = []
    decision_total = 0
    for path, summary in zip(shards, summaries, strict=True):
        selected_paths.append(path)
        selected_summaries.append(summary)
        decision_total += int(summary["decisions"])
        if decision_total >= min_decisions:
            return selected_paths, selected_summaries, decision_total
    return [], [], decision_total


def deploy(
    *,
    args: argparse.Namespace,
    league: dict[str, Any],
    chain_name: str,
    generation: int,
    checkpoint: Path,
    generation_root: Path,
    env: dict[str, str],
) -> Path:
    chain = league["chains"][chain_name]
    deployment = generation_root / "deployment"
    portable = deployment / "universal_ppo.npz"
    deck_receipt = deployment / "deck.json"
    packages = deployment / "packages"
    deployment.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        deck_receipt,
        {
            "schemaVersion": 1,
            "selected": [
                {
                    "name": chain["deckName"],
                    "archetypeId": chain["archetypeId"],
                    "archetypeLabel": chain["archetypeLabel"],
                    "deckPath": chain["deckPath"],
                    "deckSha256": chain["deckSha256"],
                }
            ],
        },
    )
    deck_input = deck_receipt
    cohort_payload = None
    if chain.get("learnerDeckPool"):
        cohort_path = deployment / "deck-cohort.json"
        cohort_payload = materialize_cohort(
            Path(chain["learnerDeckPool"]),
            cohort_path,
            size=int(chain.get("learnerDeckCohortSize", 20)),
            chain_name=chain_name,
            generation=generation,
            snapshot_sha=sha256_file(checkpoint),
        )
        deck_input = cohort_path
    tool = args.worktree / "experiment7/integration/export_and_package.py"
    export_output = portable
    staging = None
    if args.deployment_staging_root is not None:
        staging = (
            args.deployment_staging_root
            / f"{chain_name}-g{generation:06d}-{os.getpid()}-{time.time_ns()}"
        )
        staging.mkdir(parents=True, exist_ok=False)
        export_output = staging / "universal_ppo.npz"
    try:
        subprocess.run(
            [
                args.python,
                str(tool),
                "export",
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(export_output),
            ],
            check=True,
            env=env,
            cwd=args.worktree,
        )
        if export_output != portable:
            temporary = portable.with_name(
                f".{portable.name}.tmp-{os.getpid()}-{time.time_ns()}"
            )
            try:
                shutil.copyfile(export_output, temporary)
                os.replace(temporary, portable)
            finally:
                temporary.unlink(missing_ok=True)
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    subprocess.run(
        [
            args.python,
            str(tool),
            "package-universal",
            "--reference-root",
            str(
                args.reference_root_override
                if args.reference_root_override is not None
                else Path(league["referenceRoot"])
            ),
            "--sources",
            str(Path(league["sources"])),
            "--decks",
            str(deck_input),
            "--portable",
            str(portable),
            "--output-root",
            str(packages),
            "--name-prefix",
            f"live_{chain_name}_g{generation:06d}",
        ],
        check=True,
        env=env,
        cwd=args.worktree,
    )
    package_manifest = packages / "packages.json"
    if cohort_payload is not None:
        manifest_payload = read_json(package_manifest)
        manifest_payload["deckCohortReceipt"] = str(deck_input.resolve())
        manifest_payload["deckCohort"] = {
            key: value for key, value in cohort_payload.items() if key != "selected"
        }
        atomic_write_json(package_manifest, manifest_payload)
    return package_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously train and publish one asynchronous PPO chain")
    parser.add_argument("--league", type=Path, required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--buffer-root", type=Path, required=True)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-behavior-lag", type=int, default=2)
    parser.add_argument("--min-decisions", type=int, default=1)
    parser.add_argument("--teacher-anchor-coefficient", type=float, default=0.02)
    parser.add_argument("--seat1-weight", type=float, default=1.0)
    parser.add_argument("--normalize-advantages-by-player", action="store_true")
    parser.add_argument("--balance-player-minibatches", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--bootstrap-deployment", action="store_true")
    parser.add_argument(
        "--reference-root-override",
        type=Path,
        help="optional node-local reference tree used for training and packaging",
    )
    parser.add_argument(
        "--deployment-staging-root",
        type=Path,
        help="optional node-local directory used to stage compressed portable exports",
    )
    args = parser.parse_args()
    ledger_path = args.run_root / args.chain / "rollout-ledger.json"
    env = dict(os.environ)
    env.update({"PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    while True:
        league = read_json(args.league)
        if args.chain not in league["chains"]:
            raise KeyError(args.chain)
        chain = league["chains"][args.chain]
        current = chain["current"]
        learner_control = chain.get("trainingControl", {}).get("learner", {})
        min_decisions = int(learner_control.get("minDecisions", args.min_decisions))
        max_behavior_lag = int(
            learner_control.get("maxBehaviorLag", args.max_behavior_lag)
        )
        teacher_anchor = float(
            learner_control.get(
                "teacherAnchorCoefficient", args.teacher_anchor_coefficient
            )
        )
        seat1_weight = float(learner_control.get("seat1Weight", args.seat1_weight))
        learning_rate = float(learner_control.get("learningRate", 1e-5))
        ppo_epochs = int(learner_control.get("ppoEpochs", 1))
        normalize_by_player = bool(
            learner_control.get(
                "normalizeAdvantagesByPlayer", args.normalize_advantages_by_player
            )
        )
        balance_player_minibatches = bool(
            learner_control.get(
                "balancePlayerMinibatches", args.balance_player_minibatches
            )
        )
        if args.bootstrap_deployment and not current.get("packageManifest"):
            generation = int(current["generation"])
            root = args.run_root / args.chain / f"generation-{generation:06d}-bootstrap"
            manifest = deploy(
                args=args,
                league=league,
                chain_name=args.chain,
                generation=generation,
                checkpoint=Path(current["checkpoint"]),
                generation_root=root,
                env=env,
            )
            publish_snapshot(
                args.league,
                args.chain,
                generation,
                Path(current["checkpoint"]),
                manifest,
            )
            args.bootstrap_deployment = False
            continue
        ledger = read_ledger(ledger_path)
        shards, summaries = eligible_shards(
            league, args.chain, args.buffer_root, ledger, max_behavior_lag
        )
        atomic_write_json(ledger_path, ledger)
        selected_paths, selected_summaries, decision_total = select_minimum_batch(
            shards, summaries, min_decisions
        )
        if not selected_paths:
            time.sleep(args.poll_seconds)
            continue
        parent_generation = int(current["generation"])
        generation = parent_generation + 1
        generation_root = args.run_root / args.chain / f"generation-{generation:06d}"
        if generation_root.exists():
            generation_root = args.run_root / args.chain / f"generation-{generation:06d}-{time.time_ns()}"
        generation_root.mkdir(parents=True)
        allowed = {}
        for summary in selected_summaries:
            allowed[str(summary["behaviorCheckpoint"]["sha256"])] = int(summary["behaviorGeneration"])
        behavior_manifest = generation_root / "behavior-manifest.json"
        atomic_write_json(
            behavior_manifest,
            {"schemaVersion": 1, "snapshots": [{"sha256": key, "generation": value} for key, value in allowed.items()]},
        )
        batch = {
            "createdAt": utc_now(),
            "chain": args.chain,
            "parent": current,
            "rollouts": [str(path.resolve()) for path in selected_paths],
            "decisions": decision_total,
            "maxBehaviorLag": max_behavior_lag,
            "trainingControl": learner_control,
        }
        atomic_write_json(generation_root / "batch.json", batch)
        checkpoint = generation_root / "checkpoint.pt"
        metrics = generation_root / "metrics.json"
        command = [
            args.python,
            str(args.worktree / "experiment7/integration/train_universal_ppo.py"),
            "--reference-root",
            str(
                args.reference_root_override
                if args.reference_root_override is not None
                else Path(league["referenceRoot"])
            ),
            "--rollouts",
            *[str(path) for path in selected_paths],
            "--initialize-from",
            str(Path(current["checkpoint"])),
            "--teacher",
            str(Path(chain["teacher"])),
            "--output",
            str(checkpoint),
            "--metrics-output",
            str(metrics),
            "--generation",
            str(generation),
            "--current-generation",
            str(parent_generation),
            "--allowed-behavior-manifest",
            str(behavior_manifest),
            "--max-behavior-lag",
            str(max_behavior_lag),
            "--role",
            "diversity",
            "--seed",
            str((time.time_ns() ^ generation) & 0x7FFFFFFF),
            "--ppo-epochs",
            str(ppo_epochs),
            "--batch-size",
            "128",
            "--learning-rate",
            str(learning_rate),
            "--target-kl",
            "0.03",
            "--teacher-anchor-coefficient",
            str(teacher_anchor),
            "--seat1-weight",
            str(seat1_weight),
            "--max-initial-clip-fraction",
            "0.5",
            "--device",
            args.device,
        ]
        if normalize_by_player:
            command.append("--normalize-advantages-by-player")
        if balance_player_minibatches:
            command.append("--balance-player-minibatches")
        log = generation_root / "train.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
        ledger = read_ledger(ledger_path)
        destination = "consumed" if completed.returncode == 0 else "rejected"
        for path in selected_paths:
            ledger.setdefault(destination, {})[str(path.resolve())] = {
                "generation": generation,
                "recordedAt": utc_now(),
                "reason": None if completed.returncode == 0 else "trainer_failed_or_policy_shift_gate",
            }
        atomic_write_json(ledger_path, ledger)
        if completed.returncode:
            atomic_write_json(
                generation_root / "FAILED.json",
                {"createdAt": utc_now(), "returnCode": completed.returncode, "log": str(log.resolve())},
            )
            continue
        manifest = deploy(
            args=args,
            league=league,
            chain_name=args.chain,
            generation=generation,
            checkpoint=checkpoint,
            generation_root=generation_root,
            env=env,
        )
        snapshot = publish_snapshot(args.league, args.chain, generation, checkpoint, manifest)
        atomic_write_json(generation_root / "PUBLISHED.json", snapshot)


if __name__ == "__main__":
    main()
