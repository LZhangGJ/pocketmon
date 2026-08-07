from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import random
import sys
import time
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.counterfactual import counterfactual_action_values
from rl.features import history_features
from rl.ppo import load_checkpoint, model_row_for_model, sample_action, sha256_file


def read_deck(path: Path, attempts: int = 20, retry_seconds: float = 0.1) -> list[int]:
    """Read a complete deck despite concurrent public-agent refreshes."""

    last_error = "unknown"
    for attempt in range(attempts):
        try:
            deck = [
                int(line.strip())
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(deck) == 60:
                return deck
            last_error = f"observed {len(deck)} cards"
        except (FileNotFoundError, OSError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < attempts:
            time.sleep(retry_seconds)
    raise ValueError(f"deck must contain exactly 60 cards after {attempts} attempts: {path} ({last_error})")


def load_agent(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path / "main.py")
    if spec is None or spec.loader is None:
        raise ImportError(path)
    previous = Path.cwd()
    inserted = str(path) not in sys.path
    try:
        os.chdir(path)
        if inserted:
            sys.path.insert(0, str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if inserted and str(path) in sys.path:
            sys.path.remove(str(path))
        os.chdir(previous)


def install_cg(cg_dir: Path) -> None:
    package = types.ModuleType("cg")
    package.__path__ = [str(cg_dir)]
    package.__package__ = "cg"
    sys.modules["cg"] = package


def resolve_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("agents", []) if isinstance(payload, dict) else payload
    accepted = []
    for item in items:
        if item.get("status", "accepted") != "accepted":
            continue
        agent_dir = Path(item.get("agent_dir") or item.get("path") or "")
        if not agent_dir.is_absolute():
            agent_dir = (ROOT / agent_dir).resolve()
        if not (agent_dir / "main.py").is_file() or not (agent_dir / "deck.csv").is_file():
            raise FileNotFoundError(f"invalid rollout opponent: {item}")
        name = str(item["name"])
        role = str(item.get("league_role") or (
            "population" if name.startswith("population_") else "public"
        ))
        if role not in {"population", "public"}:
            raise ValueError(f"unsupported rollout opponent role {role!r}: {item}")
        # Freeze the deck once per shard. Daily notebook refreshes may replace
        # public agent artifacts while a 64-episode shard is running.
        accepted.append({
            "name": name,
            "agent_dir": agent_dir,
            "league_role": role,
            "deck": read_deck(agent_dir / "deck.csv"),
        })
    if not accepted:
        raise ValueError("rollout opponent manifest is empty")
    return accepted


def choose_frozen_opponent(
    pool: list[dict[str, Any]], rng: random.Random, frozen_league_fraction: float,
) -> dict[str, Any]:
    """Choose a frozen opponent; never train both sides of a zero-sum PPO game."""

    if not 0.0 <= frozen_league_fraction <= 1.0:
        raise ValueError("frozen league fraction must be in [0, 1]")
    population = [item for item in pool if item.get("league_role") == "population"]
    public = [item for item in pool if item.get("league_role") == "public"]
    prefer_population = rng.random() < frozen_league_fraction
    candidates = population if prefer_population else public
    if not candidates:
        candidates = public if prefer_population else population
    if not candidates:
        raise ValueError("rollout opponent pool has no frozen opponents")
    return rng.choice(candidates)


def finish_trajectory(rows: list[dict[str, Any]], winner: int) -> None:
    by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        player = int(row["player"])
        outcome = 0.0 if winner == 2 else (1.0 if winner == player else -1.0)
        row["outcome"] = outcome
        row["reward"] = 0.0
        by_player[player].append(row)
    for group in by_player.values():
        group[-1]["reward"] = float(group[-1]["outcome"])


def finish_counterfactual_rows(rows: list[dict[str, Any]], winner: int) -> None:
    """Mark real losses without changing the engine-derived Q target."""

    for row in rows:
        player = int(row["player"])
        outcome = 0.0 if winner == 2 else (1.0 if winner == player else -1.0)
        row["root_outcome"] = outcome
        row["loss_priority"] = outcome < 0.0


def play_episode(
    *,
    model,
    learner_deck: list[int],
    opponent: dict[str, Any] | None,
    learner_seat: int,
    cg_dir: Path,
    episode: int,
    episode_id: str,
    checkpoint_sha256: str,
    device: torch.device,
    temperature: float,
    max_decisions: int,
    basic_ids: set[int],
    counterfactual_rng: random.Random,
    counterfactual_rate: float,
    counterfactual_candidates: int,
    counterfactual_determinizations: int,
    counterfactual_horizon: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, str, int]:
    install_cg(cg_dir)
    from cg.game import battle_finish, battle_select, battle_start

    if opponent is None:
        raise ValueError("PPO rollouts require a frozen opponent; symmetric two-sided PPO is disabled")
    # The learner alternates seats across episodes, but the opponent is always
    # frozen. Training both sides of a shared-policy zero-sum game optimizes an
    # expected R0 + R1 == 0 objective and only adds gradient noise.
    trainable_seats = {learner_seat}
    opponent_name = str(opponent["name"])
    opponent_role = str(opponent.get("league_role", "public"))
    frozen_self_play = opponent_role == "population"
    opponent_dir = Path(opponent["agent_dir"])
    opponent_module = load_agent(opponent_dir, f"ppo_opponent_{episode}")
    opponent_deck = list(opponent.get("deck") or read_deck(opponent_dir / "deck.csv"))
    decks = [list(learner_deck), list(learner_deck)]
    decks[1 - learner_seat] = opponent_deck
    observation, start = battle_start(*decks)
    if observation is None:
        raise RuntimeError(f"battle start failed: {start.errorType}")
    trajectory: list[dict[str, Any]] = []
    counterfactual_rows: list[dict[str, Any]] = []
    counterfactual_errors = 0
    learner_history: list[list[float]] = []
    try:
        for step in range(max_decisions):
            current = observation.get("current") or {}
            result = int(current.get("result", -1))
            if result != -1:
                finish_trajectory(trajectory, result)
                finish_counterfactual_rows(counterfactual_rows, result)
                return trajectory, counterfactual_rows, result, opponent_name, counterfactual_errors
            player = int(current.get("yourIndex", step % 2))
            select = observation.get("select")
            if not isinstance(select, dict) or not isinstance(select.get("option"), list):
                raise ValueError("engine returned a non-selection observation during PPO rollout")
            if player in trainable_seats:
                if counterfactual_rate > 0.0 and counterfactual_rng.random() < counterfactual_rate:
                    try:
                        labels = counterfactual_action_values(
                            model=model,
                            observation=observation,
                            decks=decks,
                            basic_ids=basic_ids,
                            device=device,
                            rng=counterfactual_rng,
                            candidates=counterfactual_candidates,
                            determinizations=counterfactual_determinizations,
                            horizon=counterfactual_horizon,
                            history=learner_history,
                        )
                        for label in labels:
                            option_index = int(label["option_index"])
                            q_row = model_row_for_model(
                                model, observation, learner_deck, [option_index], learner_history
                            )
                            q_row.update({
                                "schema_version": 1,
                                "rollout_format": "counterfactual_action_q_v1",
                                "episode_id": episode_id,
                                "episode": episode,
                                "observation_step": step,
                                "player": player,
                                "option_index": option_index,
                                "q_target": float(label["target"]),
                                "q_target_std": float(label["target_std"]),
                                "q_samples": int(label["samples"]),
                                "behavior_checkpoint_sha256": checkpoint_sha256,
                                "opponent": opponent_name,
                                "self_play": frozen_self_play,
                                "symmetric_self_play": False,
                                "frozen_opponent": True,
                                "opponent_role": opponent_role,
                            })
                            counterfactual_rows.append(q_row)
                    except Exception:
                        counterfactual_errors += 1
                action, log_probability, value, entropy = sample_action(
                    model, observation, learner_deck, device, temperature, history=learner_history
                )
                row = model_row_for_model(
                    model, observation, learner_deck, action, learner_history
                )
                row.update({
                    "schema_version": 3,
                    "rollout_format": "masked_ppo_v1",
                    "episode_id": episode_id,
                    "episode": episode,
                    "action_step": step + 1,
                    "observation_step": step,
                    "player": player,
                    "behavior_log_probability": log_probability,
                    "behavior_value": value,
                    "behavior_entropy": entropy,
                    "behavior_checkpoint_sha256": checkpoint_sha256,
                    "temperature": temperature,
                    "trainable": True,
                    "opponent": opponent_name,
                    "self_play": frozen_self_play,
                    "symmetric_self_play": False,
                    "frozen_opponent": True,
                    "opponent_role": opponent_role,
                    "reward": 0.0,
                    "outcome": 0.0,
                })
                trajectory.append(row)
                learner_history.append(history_features(row["state"], row["options"], action))
                maximum = int(getattr(model, "history_length", 0) or 0)
                if maximum:
                    learner_history = learner_history[-maximum:]
            else:
                if opponent_module is None:
                    raise RuntimeError("missing rollout opponent module")
                action = opponent_module.agent(observation)
            observation = battle_select(action)
        raise TimeoutError(f"episode {episode_id} exceeded {max_decisions} decisions")
    finally:
        battle_finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect learner-only PPO rollouts against a frozen league")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--cg-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--frozen-league-fraction", type=float)
    parser.add_argument(
        "--self-play-fraction", type=float,
        help="Deprecated alias for --frozen-league-fraction; self-play opponents are frozen",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-decisions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counterfactual-output", type=Path)
    parser.add_argument("--counterfactual-rate", type=float, default=0.0)
    parser.add_argument("--counterfactual-candidates", type=int, default=4)
    parser.add_argument("--counterfactual-determinizations", type=int, default=2)
    parser.add_argument("--counterfactual-horizon", type=int, default=16)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    frozen_league_fraction = (
        args.frozen_league_fraction
        if args.frozen_league_fraction is not None
        else (args.self_play_fraction if args.self_play_fraction is not None else 0.5)
    )
    if not 0.0 <= frozen_league_fraction <= 1.0:
        raise ValueError("frozen league fraction must be in [0, 1]")
    if not 0.0 <= args.counterfactual_rate <= 1.0:
        raise ValueError("counterfactual rate must be in [0, 1]")
    if args.counterfactual_rate and args.counterfactual_output is None:
        raise ValueError("counterfactual output is required when counterfactual sampling is enabled")
    if min(args.counterfactual_candidates, args.counterfactual_determinizations, args.counterfactual_horizon) <= 0:
        raise ValueError("counterfactual search parameters must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite rollout shard: {args.output}")
    if args.counterfactual_output and args.counterfactual_output.exists():
        raise FileExistsError(f"refusing to overwrite counterfactual shard: {args.counterfactual_output}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    model.eval()
    checkpoint_sha256 = sha256_file(args.checkpoint)
    learner_deck = read_deck(args.deck)
    pool = resolve_manifest(args.pool)
    cg_dir = args.cg_dir.resolve()
    install_cg(cg_dir)
    from cg.api import all_card_data
    basic_ids = {int(card.cardId) for card in all_card_data() if bool(card.basic)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    counterfactual_temporary = (
        args.counterfactual_output.with_name(args.counterfactual_output.name + ".tmp")
        if args.counterfactual_output else None
    )
    counters = Counter()
    opponent_counts = Counter()
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle, (
            gzip.open(counterfactual_temporary, "wt", encoding="utf-8")
            if counterfactual_temporary else open(os.devnull, "w", encoding="utf-8")
        ) as counterfactual_handle:
            for episode in range(args.episodes):
                opponent = choose_frozen_opponent(pool, rng, frozen_league_fraction)
                frozen_self_play = opponent["league_role"] == "population"
                learner_seat = episode % 2
                episode_id = f"{args.run_id}-{episode:06d}"
                rows, q_rows, winner, opponent_name, q_errors = play_episode(
                    model=model,
                    learner_deck=learner_deck,
                    opponent=opponent,
                    learner_seat=learner_seat,
                    cg_dir=cg_dir,
                    episode=episode,
                    episode_id=episode_id,
                    checkpoint_sha256=checkpoint_sha256,
                    device=device,
                    temperature=args.temperature,
                    max_decisions=args.max_decisions,
                    basic_ids=basic_ids,
                    counterfactual_rng=rng,
                    counterfactual_rate=args.counterfactual_rate,
                    counterfactual_candidates=args.counterfactual_candidates,
                    counterfactual_determinizations=args.counterfactual_determinizations,
                    counterfactual_horizon=args.counterfactual_horizon,
                )
                for row in rows:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                for row in q_rows:
                    counterfactual_handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                counters["episodes"] += 1
                counters["rows"] += len(rows)
                counters["counterfactual_rows"] += len(q_rows)
                counters["counterfactual_errors"] += q_errors
                counters["frozen_self_play_episodes"] += int(frozen_self_play)
                counters["public_opponent_episodes"] += int(not frozen_self_play)
                counters["symmetric_self_play_episodes"] += 0
                counters["wins"] += int(winner == learner_seat)
                counters["losses"] += int(winner in {0, 1} and winner != learner_seat)
                counters["draws"] += int(winner == 2)
                opponent_counts[opponent_name] += 1
                if counters["episodes"] % 10 == 0:
                    print(json.dumps({"progress": f"{counters['episodes']}/{args.episodes}", "rows": counters["rows"]}), flush=True)
        temporary.replace(args.output)
        if counterfactual_temporary and args.counterfactual_output:
            counterfactual_temporary.replace(args.counterfactual_output)
    except Exception:
        temporary.unlink(missing_ok=True)
        if counterfactual_temporary:
            counterfactual_temporary.unlink(missing_ok=True)
        raise
    summary = {
        **dict(counters),
        "run_id": args.run_id,
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "behavior_checkpoint": str(args.checkpoint),
        "behavior_checkpoint_sha256": checkpoint_sha256,
        "engine_seed_controlled": False,
        "temperature": args.temperature,
        "frozen_league_fraction": frozen_league_fraction,
        "symmetric_self_play_fraction": 0.0,
        "opponents": dict(opponent_counts),
        "counterfactual_output": str(args.counterfactual_output) if args.counterfactual_output else None,
        "counterfactual_output_sha256": (
            sha256_file(args.counterfactual_output) if args.counterfactual_output else None
        ),
        "counterfactual_rate": args.counterfactual_rate,
        "counterfactual_candidates": args.counterfactual_candidates,
        "counterfactual_determinizations": args.counterfactual_determinizations,
        "counterfactual_horizon": args.counterfactual_horizon,
    }
    summary_path = args.output.with_name(args.output.name + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
