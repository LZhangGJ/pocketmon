from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import random
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.ppo import load_checkpoint, model_row_from_observation, sample_action, sha256_file


def read_deck(path: Path) -> list[int]:
    deck = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"deck must contain exactly 60 cards: {path}")
    return deck


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
        accepted.append({"name": str(item["name"]), "agent_dir": agent_dir})
    if not accepted:
        raise ValueError("rollout opponent manifest is empty")
    return accepted


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
) -> tuple[list[dict[str, Any]], int, str]:
    install_cg(cg_dir)
    from cg.game import battle_finish, battle_select, battle_start

    self_play = opponent is None
    trainable_seats = {0, 1} if self_play else {learner_seat}
    opponent_module = None
    opponent_deck: list[int] | None = None
    opponent_name = "self_play"
    if opponent is not None:
        opponent_name = str(opponent["name"])
        opponent_dir = Path(opponent["agent_dir"])
        opponent_module = load_agent(opponent_dir, f"ppo_opponent_{episode}")
        opponent_deck = read_deck(opponent_dir / "deck.csv")
    decks = [list(learner_deck), list(learner_deck)]
    if opponent_deck is not None:
        decks[1 - learner_seat] = opponent_deck
    observation, start = battle_start(*decks)
    if observation is None:
        raise RuntimeError(f"battle start failed: {start.errorType}")
    trajectory: list[dict[str, Any]] = []
    try:
        for step in range(max_decisions):
            current = observation.get("current") or {}
            result = int(current.get("result", -1))
            if result != -1:
                finish_trajectory(trajectory, result)
                return trajectory, result, opponent_name
            player = int(current.get("yourIndex", step % 2))
            select = observation.get("select")
            if not isinstance(select, dict) or not isinstance(select.get("option"), list):
                raise ValueError("engine returned a non-selection observation during PPO rollout")
            if player in trainable_seats:
                action, log_probability, value, entropy = sample_action(
                    model, observation, learner_deck, device, temperature
                )
                row = model_row_from_observation(observation, learner_deck, action)
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
                    "self_play": self_play,
                    "reward": 0.0,
                    "outcome": 0.0,
                })
                trajectory.append(row)
            else:
                if opponent_module is None:
                    raise RuntimeError("missing rollout opponent module")
                action = opponent_module.agent(observation)
            observation = battle_select(action)
        raise TimeoutError(f"episode {episode_id} exceeded {max_decisions} decisions")
    finally:
        battle_finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect stochastic on-policy masked PPO self-play rollouts")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--cg-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--self-play-fraction", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-decisions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if not 0.0 <= args.self_play_fraction <= 1.0:
        raise ValueError("self-play fraction must be in [0, 1]")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite rollout shard: {args.output}")

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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    counters = Counter()
    opponent_counts = Counter()
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            for episode in range(args.episodes):
                self_play = rng.random() < args.self_play_fraction
                opponent = None if self_play else rng.choice(pool)
                learner_seat = episode % 2
                episode_id = f"{args.run_id}-{episode:06d}"
                rows, winner, opponent_name = play_episode(
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
                )
                for row in rows:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                counters["episodes"] += 1
                counters["rows"] += len(rows)
                counters["self_play_episodes"] += int(self_play)
                counters["wins"] += int(winner in ({0, 1} if self_play else {learner_seat}))
                counters["losses"] += int(not self_play and winner in {0, 1} and winner != learner_seat)
                counters["draws"] += int(winner == 2)
                opponent_counts[opponent_name] += 1
                if counters["episodes"] % 10 == 0:
                    print(json.dumps({"progress": f"{counters['episodes']}/{args.episodes}", "rows": counters["rows"]}), flush=True)
        temporary.replace(args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
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
        "self_play_fraction": args.self_play_fraction,
        "opponents": dict(opponent_counts),
    }
    summary_path = args.output.with_name(args.output.name + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
