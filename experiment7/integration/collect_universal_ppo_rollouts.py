from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
import types
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch

from agent_isolation import call_agent, isolated_agent_workdir, load_agent
from common import Experiment7Error, sha256_file, utc_now, write_json
from universal_ppo import (
    ROLLOUT_FORMAT,
    append_history,
    collate_rows,
    evaluate_actions,
    live_row,
    load_feature_runtime,
    load_universal_checkpoint,
    sample_action,
)


def read_deck(path: Path) -> list[int]:
    deck = [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise Experiment7Error(f"deck must contain exactly 60 cards: {path}")
    return deck


def install_cg(cg_dir: Path) -> None:
    package = types.ModuleType("cg")
    package.__path__ = [str(cg_dir.resolve())]
    package.__package__ = "cg"
    sys.modules["cg"] = package


def load_pool(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("agents", []) if isinstance(payload, dict) else payload
    result = []
    for row in items:
        if row.get("status", "accepted") != "accepted":
            continue
        agent_dir = Path(row.get("agent_dir") or row.get("path") or "").resolve()
        if not (agent_dir / "main.py").is_file() or not (agent_dir / "deck.csv").is_file():
            raise FileNotFoundError(agent_dir)
        result.append({**row, "name": str(row["name"]), "agent_dir": agent_dir})
    if not result:
        raise Experiment7Error("Universal PPO opponent pool is empty")
    return result


def choose_opponent(pool: list[dict[str, Any]], role: str, rng: random.Random) -> dict[str, Any]:
    if role == "diversity":
        by_archetype: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pool:
            by_archetype[str(row.get("archetype", "unknown"))].append(row)
        return rng.choice(by_archetype[rng.choice(sorted(by_archetype))])
    if role == "hard_exploiter":
        weights = [3.0 if row.get("skill_tier") == "hard" else 1.0 for row in pool]
        return rng.choices(pool, weights=weights, k=1)[0]
    if role == "conservative":
        weights = [max(float(row.get("screening", {}).get("score_rate", 0.5)), 0.1) for row in pool]
        return rng.choices(pool, weights=weights, k=1)[0]
    return rng.choice(pool)


def finish_trajectory(rows: list[dict[str, Any]], winner: int) -> None:
    by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        player = int(row["player"])
        row["outcome"] = 0.0 if winner == 2 else (1.0 if winner == player else -1.0)
        row["reward"] = 0.0
        by_player[player].append(row)
    for group in by_player.values():
        group[-1]["reward"] = float(group[-1]["outcome"])


def play_episode(
    *,
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    runtime: Any,
    learner_deck: list[int],
    opponent: dict[str, Any] | None,
    learner_seat: int,
    cg_dir: Path,
    episode: int,
    episode_id: str,
    checkpoint_sha256: str,
    teacher_sha256: str,
    device: torch.device,
    temperature: float,
    max_decisions: int,
) -> tuple[list[dict[str, Any]], int, str]:
    install_cg(cg_dir)
    from cg.game import battle_finish, battle_select, battle_start

    self_play = opponent is None
    trainable = {0, 1} if self_play else {learner_seat}
    with ExitStack() as stack:
        opponent_module = None
        opponent_workdir = None
        opponent_name = "self_play"
        decks = [list(learner_deck), list(learner_deck)]
        if opponent is not None:
            opponent_name = str(opponent["name"])
            opponent_dir = Path(opponent["agent_dir"])
            # Lock the match deck before importing untrusted agent code.  Some
            # public agents rewrite deck.csv while importing their module.
            decks[1 - learner_seat] = read_deck(opponent_dir / "deck.csv")
            opponent_workdir = stack.enter_context(isolated_agent_workdir(opponent_dir))
            opponent_module = load_agent(
                opponent_dir, f"universal_ppo_opponent_{episode}", opponent_workdir
            )
        observation, start = battle_start(*decks)
        if observation is None:
            raise Experiment7Error(f"battle_start failed: {start.errorType}")
        rows: list[dict[str, Any]] = []
        histories: dict[int, list[Any]] = {0: [], 1: []}
        try:
            for step in range(max_decisions):
                current = observation.get("current") or {}
                result = int(current.get("result", -1))
                if result != -1:
                    finish_trajectory(rows, result)
                    return rows, result, opponent_name
                player = int(current.get("yourIndex", step % 2))
                select = observation.get("select")
                if not isinstance(select, dict) or not isinstance(select.get("option"), list):
                    raise Experiment7Error("engine returned a non-selection observation")
                if player in trainable:
                    feature_row, state, option_rows = live_row(
                        observation, decks[player], histories[player], runtime, model.config
                    )
                    option_count = len(feature_row["options"])
                    minimum = int(feature_row["min_count"])
                    maximum = int(feature_row["max_count"])
                    forced = minimum == maximum and minimum in (0, option_count)
                    if forced:
                        action = list(range(minimum))
                    else:
                        action, log_probability, value, entropy = sample_action(
                            model, feature_row, device, temperature
                        )
                        decision = {**feature_row, "action": action}
                        with torch.inference_mode():
                            teacher_log_probability, _, _ = evaluate_actions(
                                teacher, collate_rows([decision], device)
                            )
                        decision.update(
                            {
                                "schema_version": 1,
                                "rollout_format": ROLLOUT_FORMAT,
                                "episode_id": episode_id,
                                "episode": episode,
                                "action_step": step + 1,
                                "observation_step": step,
                                "player": player,
                                "behavior_log_probability": log_probability,
                                "teacher_log_probability": float(teacher_log_probability[0]),
                                "behavior_value": value,
                                "behavior_entropy": entropy,
                                "behavior_checkpoint_sha256": checkpoint_sha256,
                                "teacher_checkpoint_sha256": teacher_sha256,
                                "temperature": temperature,
                                "opponent": opponent_name,
                                "self_play": self_play,
                                "reward": 0.0,
                                "outcome": 0.0,
                            }
                        )
                        rows.append(decision)
                    append_history(
                        histories[player],
                        state,
                        option_rows,
                        action,
                        int(model.config.history_length),
                    )
                else:
                    if opponent_module is None:
                        raise Experiment7Error("missing external opponent module")
                    if opponent_workdir is None:
                        raise Experiment7Error("missing isolated opponent working directory")
                    action = call_agent(opponent_module, observation, opponent_workdir)
                observation = battle_select(action)
            raise TimeoutError(f"episode {episode_id} exceeded {max_decisions} decisions")
        finally:
            battle_finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Deck-8 Universal PPO rollouts")
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--engine-catalog", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--cg-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--self-play-fraction", type=float, default=0.25)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-decisions", type=int, default=5000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--role", choices=("generalist", "hard_exploiter", "diversity", "conservative"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.episodes <= 0 or not 0 <= args.self_play_fraction <= 1:
        raise ValueError("invalid rollout episode configuration")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, _ = load_universal_checkpoint(args.checkpoint.resolve(), args.reference_root, device)
    teacher, _ = load_universal_checkpoint(args.teacher.resolve(), args.reference_root, device)
    model.eval()
    teacher.eval()
    runtime = load_feature_runtime(args.reference_root, args.engine_catalog)
    learner_deck = read_deck(args.deck.resolve())
    pool = load_pool(args.pool.resolve())
    rng = random.Random(args.seed)
    behavior_sha = sha256_file(args.checkpoint)
    teacher_sha = sha256_file(args.teacher)
    temporary = args.output.with_name(args.output.name + ".tmp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    opponents: Counter[str] = Counter()
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for episode in range(args.episodes):
            self_play = rng.random() < args.self_play_fraction
            opponent = None if self_play else choose_opponent(pool, args.role, rng)
            rows, winner, opponent_name = play_episode(
                model=model,
                teacher=teacher,
                runtime=runtime,
                learner_deck=learner_deck,
                opponent=opponent,
                learner_seat=episode % 2,
                cg_dir=args.cg_dir.resolve(),
                episode=episode,
                episode_id=f"{args.run_id}-{episode:06d}",
                checkpoint_sha256=behavior_sha,
                teacher_sha256=teacher_sha,
                device=device,
                temperature=args.temperature,
                max_decisions=args.max_decisions,
            )
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            counters["episodes"] += 1
            counters["decisions"] += len(rows)
            counters["selfPlayEpisodes"] += int(self_play)
            counters["draws"] += int(winner == 2)
            if not self_play:
                learner_seat = episode % 2
                counters["wins"] += int(winner == learner_seat)
                counters["losses"] += int(winner in (0, 1) and winner != learner_seat)
            opponents[opponent_name] += 1
            print(json.dumps({"progress": f"{episode + 1}/{args.episodes}", "decisions": counters["decisions"]}), flush=True)
    temporary.replace(args.output)
    summary = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "rolloutFormat": ROLLOUT_FORMAT,
        "role": args.role,
        **dict(counters),
        "opponents": dict(opponents),
        "behaviorCheckpoint": {"path": str(args.checkpoint.resolve()), "sha256": behavior_sha},
        "teacherCheckpoint": {"path": str(args.teacher.resolve()), "sha256": teacher_sha},
        "output": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output)},
        "engineSeedControlled": False,
    }
    write_json(args.output.with_suffix(args.output.suffix + ".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
