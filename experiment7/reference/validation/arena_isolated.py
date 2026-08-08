#!/usr/bin/env python3
"""Seat-balanced local arena with each stateful agent isolated in its own process."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any


def load_agent(name: str, root: Path) -> ModuleType:
    sys.path.insert(0, str(root.resolve()))
    spec = importlib.util.spec_from_file_location(name, root / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import agent from {root}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def agent_worker(connection: Any, name: str, root_text: str) -> None:
    try:
        root = Path(root_text)
        os.chdir(root)
        module = load_agent(name, root)
        connection.send(("ready", None))
        while True:
            command, payload = connection.recv()
            if command == "stop":
                connection.send(("stopped", None))
                return
            if command == "act":
                try:
                    action = list(module.agent(payload))
                    connection.send(("ok", action))
                except Exception as exc:
                    connection.send(("error", f"{type(exc).__name__}: {exc}"))
            elif command == "stats":
                advisor = getattr(module, "bc_advisor", None)
                stats = advisor.get_stats() if advisor is not None and hasattr(advisor, "get_stats") else {}
                connection.send(("ok", stats))
            else:
                connection.send(("error", f"unknown command {command}"))
    except Exception as exc:
        connection.send(("fatal", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


class AgentProcess:
    def __init__(self, context: Any, name: str, root: Path):
        parent, child = context.Pipe()
        self.connection = parent
        self.process = context.Process(target=agent_worker, args=(child, name, str(root)))
        self.process.start()
        status, payload = self.connection.recv()
        if status != "ready":
            raise RuntimeError(f"agent {name} failed to start: {status} {payload}")

    def act(self, observation: dict[str, Any]) -> list[int]:
        self.connection.send(("act", observation))
        status, payload = self.connection.recv()
        if status != "ok":
            raise RuntimeError(f"agent call failed: {status} {payload}")
        return list(payload)

    def stats(self) -> dict[str, Any]:
        self.connection.send(("stats", None))
        status, payload = self.connection.recv()
        if status != "ok":
            raise RuntimeError(f"agent stats failed: {status} {payload}")
        return dict(payload)

    def close(self) -> None:
        if self.process.is_alive():
            self.connection.send(("stop", None))
            try:
                self.connection.recv()
            except (EOFError, BrokenPipeError):
                pass
        self.connection.close()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)


def directory_receipt(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and "__pycache__" not in item.parts):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def validate_action(action: list[int], observation: dict[str, Any]) -> None:
    select = observation.get("select") or {}
    options = select.get("option") or []
    minimum = int(select.get("minCount", 0) or 0)
    maximum = int(select.get("maxCount", 0) or 0)
    if not (
        isinstance(action, list)
        and all(isinstance(index, int) and not isinstance(index, bool) for index in action)
        and len(action) == len(set(action))
        and minimum <= len(action) <= maximum
        and all(0 <= index < len(options) for index in action)
    ):
        raise ValueError(
            f"illegal action={action}, min={minimum}, max={maximum}, options={len(options)}"
        )


def wilson_interval(wins: int, losses: int, draws: int, z: float = 1.96) -> tuple[float, float]:
    n = wins + losses + draws
    if n == 0:
        return 0.0, 0.0
    score = (wins + 0.5 * draws) / n
    denominator = 1.0 + z * z / n
    center = (score + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(score * (1.0 - score) / n + z * z / (4.0 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--engine-root", type=Path)
    parser.add_argument("--games-per-seat", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--baseline-label", default="baseline")
    args = parser.parse_args()

    engine_root = args.engine_root or args.candidate_root
    sys.path.insert(0, str(engine_root.resolve()))
    from cg.game import battle_finish, battle_select, battle_start

    context = mp.get_context("spawn")
    candidate = AgentProcess(context, "isolated_candidate_main", args.candidate_root)
    baseline = AgentProcess(context, "isolated_baseline_main", args.baseline_root)
    try:
        reset_observation = {"select": None, "logs": [], "current": None}
        candidate_deck = candidate.act(reset_observation)
        baseline_deck = baseline.act(reset_observation)
        if len(candidate_deck) != 60 or len(baseline_deck) != 60:
            raise ValueError("both agents must return a 60-card deck")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        game_rows: list[dict[str, object]] = []
        for candidate_seat in (0, 1):
            for game_index in range(args.games_per_seat):
                if candidate.act(reset_observation) != candidate_deck:
                    raise RuntimeError("candidate deck changed between games")
                if baseline.act(reset_observation) != baseline_deck:
                    raise RuntimeError("baseline deck changed between games")
                agents = (candidate, baseline) if candidate_seat == 0 else (baseline, candidate)
                decks = (
                    (candidate_deck, baseline_deck)
                    if candidate_seat == 0
                    else (baseline_deck, candidate_deck)
                )
                started = time.perf_counter()
                decisions = 0
                error = ""
                result = -1
                observation = None
                try:
                    observation, start_data = battle_start(decks[0], decks[1])
                    if observation is None:
                        raise RuntimeError(
                            f"battle_start failed: player={start_data.errorPlayer}, type={start_data.errorType}"
                        )
                    while decisions < args.max_decisions:
                        current = observation.get("current") or {}
                        result = int(current.get("result", -1))
                        if result >= 0:
                            break
                        player = int(current.get("yourIndex", -1))
                        if player not in (0, 1):
                            raise RuntimeError(f"invalid selecting player {player}")
                        action = agents[player].act(observation)
                        validate_action(action, observation)
                        observation = battle_select(action)
                        decisions += 1
                    else:
                        raise TimeoutError(f"exceeded {args.max_decisions} decisions")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    try:
                        battle_finish()
                    except Exception:
                        pass
                candidate_result = (
                    "error"
                    if error
                    else "draw"
                    if result == 2
                    else "win"
                    if result == candidate_seat
                    else "loss"
                )
                row = {
                    "candidate_seat": candidate_seat,
                    "game_index": game_index,
                    "engine_result": result,
                    "candidate_result": candidate_result,
                    "decisions": decisions,
                    "elapsed_seconds": time.perf_counter() - started,
                    "error": error,
                }
                game_rows.append(row)
                print(
                    json.dumps(
                        {
                            "completed": len(game_rows),
                            "candidateSeat": candidate_seat,
                            "candidateResult": candidate_result,
                            "decisions": decisions,
                            "seconds": round(float(row["elapsed_seconds"]), 3),
                            "error": error,
                        }
                    ),
                    flush=True,
                )

        with (args.output_dir / "games.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(game_rows[0]))
            writer.writeheader()
            writer.writerows(game_rows)
        wins = sum(row["candidate_result"] == "win" for row in game_rows)
        losses = sum(row["candidate_result"] == "loss" for row in game_rows)
        draws = sum(row["candidate_result"] == "draw" for row in game_rows)
        errors = sum(bool(row["error"]) for row in game_rows)
        low, high = wilson_interval(wins, losses, draws)
        summary = {
            "candidate": args.candidate_label,
            "baseline": args.baseline_label,
            "processIsolatedAgents": True,
            "sameDeck": candidate_deck == baseline_deck,
            "games": len(game_rows),
            "gamesPerSeat": args.games_per_seat,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "errors": errors,
            "candidateScoreRate": (wins + 0.5 * draws) / max(1, wins + losses + draws),
            "wilson95": [low, high],
            "candidateAgentStats": candidate.stats(),
            "candidatePackageSha256": directory_receipt(args.candidate_root),
            "baselinePackageSha256": directory_receipt(args.baseline_root),
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return 1 if errors else 0
    finally:
        candidate.close()
        baseline.close()


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
