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
from ppo_tactical_shaping import (
    TACTICAL_ERROR_TO_OPPORTUNITY,
    TacticalShapingState,
    finalize_tactical_audit_row,
    tactical_adjustment,
    tactical_hard_mask_options,
    tactical_search_logit_biases,
)


CANONICAL_ARCHETYPE_ALIASES = {
    "alakazam": "A03",
    "grimmsnarl_froslass_munkidori": "A02",
    "dragapult": "A06",
    "mega_lucario": "LUCARIO",
    "mega_lucario_ex": "LUCARIO",
}


def canonical_archetype(row: dict[str, Any]) -> str:
    """Return the B08 sampling bucket, collapsing known alias tags."""
    explicit = row.get("canonical_archetype")
    if explicit:
        return str(explicit).upper()
    archetype = str(row.get("archetype", "unknown"))
    return CANONICAL_ARCHETYPE_ALIASES.get(archetype.lower(), archetype.upper())


def read_deck(path: Path) -> list[int]:
    deck = [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(deck) != 60:
        raise Experiment7Error(f"deck must contain exactly 60 cards: {path}")
    return deck


def load_learner_deck_pool(path: Path | None, fallback: Path) -> list[dict[str, Any]]:
    if path is None:
        return [
            {
                "name": fallback.stem,
                "path": fallback.resolve(),
                "deck": read_deck(fallback.resolve()),
                "weight": 1.0,
            }
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("selected", []) if isinstance(payload, dict) else []
    if not items:
        raise Experiment7Error(f"learner deck pool is empty: {path}")
    result: list[dict[str, Any]] = []
    for row in items:
        deck_path = Path(row.get("deckPath") or row.get("path") or "")
        if not deck_path.is_absolute():
            deck_path = (path.parent / deck_path).resolve()
        weight = float(row.get("samplingWeight", row.get("sampling_weight", 1.0)))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"invalid learner deck weight: {row}")
        result.append(
            {
                "name": str(row.get("name") or row.get("deckSha256") or deck_path.stem),
                "path": deck_path,
                "weight": weight,
                "deck": read_deck(deck_path),
            }
        )
    return result


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


def positive_weights(payload: str | None) -> dict[str, float]:
    if not payload:
        return {}
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise ValueError("sampling weights must be a JSON object")
    result = {}
    for key, value in raw.items():
        weight = float(value)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"invalid sampling weight for {key}: {value}")
        result[str(key)] = weight
    return result


def choose_opponent(
    pool: list[dict[str, Any]],
    role: str,
    rng: random.Random,
    archetype_weights: dict[str, float] | None = None,
    agent_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    archetype_weights = archetype_weights or {}
    agent_weights = agent_weights or {}
    if role == "diversity":
        by_archetype: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pool:
            by_archetype[canonical_archetype(row)].append(row)
        archetypes = sorted(by_archetype)
        archetype = rng.choices(
            archetypes,
            weights=[archetype_weights.get(name, 1.0) for name in archetypes],
            k=1,
        )[0]
        candidates = by_archetype[archetype]
        return rng.choices(
            candidates,
            weights=[agent_weights.get(str(row["name"]), 1.0) for row in candidates],
            k=1,
        )[0]
    if role == "hard_exploiter":
        weights = [
            (3.0 if row.get("skill_tier") == "hard" else 1.0)
            * archetype_weights.get(canonical_archetype(row), 1.0)
            * agent_weights.get(str(row["name"]), 1.0)
            for row in pool
        ]
        return rng.choices(pool, weights=weights, k=1)[0]
    if role == "conservative":
        weights = [
            max(float(row.get("screening", {}).get("score_rate", 0.5)), 0.1)
            * archetype_weights.get(canonical_archetype(row), 1.0)
            * agent_weights.get(str(row["name"]), 1.0)
            for row in pool
        ]
        return rng.choices(pool, weights=weights, k=1)[0]
    return rng.choices(
        pool,
        weights=[
            archetype_weights.get(canonical_archetype(row), 1.0)
            * agent_weights.get(str(row["name"]), 1.0)
            for row in pool
        ],
        k=1,
    )[0]


def _player_state(observation: dict[str, Any], player: int) -> dict[str, Any]:
    players = (observation.get("current") or {}).get("players") or []
    if 0 <= player < len(players) and isinstance(players[player], dict):
        return players[player]
    return {}


def _prize_count(observation: dict[str, Any], player: int) -> int:
    return len(_player_state(observation, player).get("prize") or [])


def _active_card(observation: dict[str, Any], player: int) -> dict[str, Any] | None:
    active = _player_state(observation, player).get("active") or []
    return active[0] if active and isinstance(active[0], dict) else None


def _start_attack_audit(
    row: dict[str, Any], observation: dict[str, Any], player: int
) -> dict[str, Any]:
    opponent_active = _active_card(observation, 1 - player) or {}
    return {
        "row": row,
        "player": player,
        "turn": int((observation.get("current") or {}).get("turn", 0) or 0),
        "prizes_before": _prize_count(observation, player),
        "opponent_active_id": int(opponent_active.get("id", 0) or 0),
        "opponent_active_hp": int(opponent_active.get("hp", 0) or 0),
    }


def _update_pending_attack_audits(
    pending: dict[int, dict[str, Any]], observation: dict[str, Any]
) -> None:
    current = observation.get("current") or {}
    turn = int(current.get("turn", 0) or 0)
    raw_result = current.get("result", -1)
    result = int(raw_result) if isinstance(raw_result, (int, float)) else -1
    completed = []
    for player, audit in pending.items():
        row = audit["row"]
        prize_delta = max(
            int(row.get("prize_delta", 0)),
            int(audit["prizes_before"]) - _prize_count(observation, player),
        )
        row["prize_delta"] = prize_delta
        opponent_active = _active_card(observation, 1 - player)
        same_active_was_knocked_out = (
            int(audit["opponent_active_hp"]) > 0
            and opponent_active is not None
            and int(opponent_active.get("id", 0) or 0)
            == int(audit["opponent_active_id"])
            and int(opponent_active.get("hp", 0) or 0) <= 0
        )
        active_disappeared = (
            int(audit["opponent_active_hp"]) > 0 and opponent_active is None
        )
        row["ko"] = bool(
            row.get("ko", False)
            or prize_delta > 0
            or same_active_was_knocked_out
            or active_disappeared
        )
        if result != -1:
            row["terminal_after_action"] = True
            completed.append(player)
        elif turn != int(audit["turn"]):
            completed.append(player)
    for player in completed:
        pending.pop(player, None)


def finish_trajectory(
    rows: list[dict[str, Any]],
    winner: int,
    *,
    long_game_min_player_decisions: int = 0,
    long_game_weight: float = 1.0,
) -> None:
    by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        player = int(row["player"])
        finalize_tactical_audit_row(row, final_win=winner == player)
        row["outcome"] = 0.0 if winner == 2 else (1.0 if winner == player else -1.0)
        row["reward"] = float(row.get("tactical_reward", 0.0))
        by_player[player].append(row)
    for group in by_player.values():
        long_game = (
            long_game_min_player_decisions > 0
            and len(group) >= long_game_min_player_decisions
        )
        for row in group:
            row["long_game_episode"] = long_game
            row["sample_weight"] = long_game_weight if long_game else 1.0
        group[-1]["reward"] += float(group[-1]["outcome"])


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
    behavior_generation: int,
    behavior_snapshot_id: str,
    teacher_sha256: str,
    device: torch.device,
    temperature: float,
    max_decisions: int,
    tactical_shaping_profile: str,
    tactical_shaping_revision: int,
    boss_reservation_penalty: float,
    boss_reservation_preference: bool,
    boss_post_play_penalty: float,
    boss_post_play_preference: bool,
    a02_poffin_decline_penalty: float,
    a02_poffin_preference: bool,
    a02_munkidori_overfill_penalty: float,
    a02_bench_budget_preference: bool,
    a02_outcome_gated_ordering: bool,
    a02_projected_bench_budget: bool,
    successor_attach_preference: bool,
    a08_terminal_before_evolve_mode: str,
    a08_gated_attack_penalty: float,
    a08_maximum_belt_support_penalty: float,
    a08_maximum_belt_preference: bool,
    a08_second_attacker_reward: float,
    a08_recovery_end_penalty: float,
    a08_recovery_preference: bool,
    end_with_attack_penalty: float,
    end_with_attack_preference: bool,
    lucario_evolve_penalty: float,
    lucario_attach_penalty: float,
    lucario_aura_overkill_penalty: float,
    lucario_aura_hard_mask: bool,
    lucario_ordering_preference: bool,
    dragapult_ready_attacker_penalty: float,
    dragapult_evolve_penalty: float,
    dragapult_wall_penalty: float,
    dragapult_budew_overstay_penalty: float,
    dragapult_resource_penalty: float,
    dragapult_wall_preference: bool,
    dragapult_terminal_search_depth: int,
    dragapult_search_bias_scale: float,
    long_game_min_player_decisions: int,
    long_game_weight: float,
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
        tactical_state = TacticalShapingState()
        pending_attack_audits: dict[int, dict[str, Any]] = {}
        try:
            for step in range(max_decisions):
                _update_pending_attack_audits(pending_attack_audits, observation)
                current = observation.get("current") or {}
                result = int(current.get("result", -1))
                if result != -1:
                    finish_trajectory(
                        rows,
                        result,
                        long_game_min_player_decisions=long_game_min_player_decisions,
                        long_game_weight=long_game_weight,
                    )
                    return rows, result, opponent_name
                player = int(current.get("yourIndex", step % 2))
                select = observation.get("select")
                if not isinstance(select, dict) or not isinstance(select.get("option"), list):
                    raise Experiment7Error("engine returned a non-selection observation")
                options = select["option"]
                recorded_decision: dict[str, Any] | None = None
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
                        option_logit_bias, search_opportunities = (
                            tactical_search_logit_biases(
                                tactical_shaping_profile,
                                observation,
                                options,
                                features=runtime.features,
                                cards=runtime.cards,
                                attacks=runtime.attacks,
                                depth=dragapult_terminal_search_depth,
                                scale=dragapult_search_bias_scale,
                            )
                        )
                        if any(option_logit_bias):
                            feature_row["option_logit_bias"] = list(
                                option_logit_bias
                            )
                        hard_masked_options = tactical_hard_mask_options(
                            tactical_shaping_profile,
                            observation,
                            options,
                            features=runtime.features,
                            cards=runtime.cards,
                            attacks=runtime.attacks,
                            lucario_aura_hard_mask=lucario_aura_hard_mask,
                        )
                        if hard_masked_options:
                            feature_row["hard_masked_options"] = list(
                                hard_masked_options
                            )
                        action, log_probability, value, entropy = sample_action(
                            model, feature_row, device, temperature
                        )
                        decision = {**feature_row, "action": action}
                        adjustment = tactical_adjustment(
                            tactical_shaping_profile,
                            observation,
                            options,
                            action,
                            features=runtime.features,
                            cards=runtime.cards,
                            attacks=runtime.attacks,
                            state=tactical_state,
                            boss_reservation_penalty=boss_reservation_penalty,
                            boss_reservation_preference=boss_reservation_preference,
                            boss_post_play_penalty=boss_post_play_penalty,
                            boss_post_play_preference=boss_post_play_preference,
                            a02_poffin_decline_penalty=a02_poffin_decline_penalty,
                            a02_poffin_preference=a02_poffin_preference,
                            a02_munkidori_overfill_penalty=(
                                a02_munkidori_overfill_penalty
                            ),
                            a02_bench_budget_preference=(
                                a02_bench_budget_preference
                            ),
                            a02_outcome_gated_ordering=a02_outcome_gated_ordering,
                            a02_projected_bench_budget=a02_projected_bench_budget,
                            successor_attach_preference=successor_attach_preference,
                            a08_terminal_before_evolve_mode=(
                                a08_terminal_before_evolve_mode
                            ),
                            a08_gated_attack_penalty=a08_gated_attack_penalty,
                            a08_maximum_belt_support_penalty=(
                                a08_maximum_belt_support_penalty
                            ),
                            a08_maximum_belt_preference=(
                                a08_maximum_belt_preference
                            ),
                            a08_second_attacker_reward=a08_second_attacker_reward,
                            a08_recovery_end_penalty=a08_recovery_end_penalty,
                            a08_recovery_preference=a08_recovery_preference,
                            end_with_attack_penalty=end_with_attack_penalty,
                            end_with_attack_preference=end_with_attack_preference,
                            lucario_evolve_penalty=lucario_evolve_penalty,
                            lucario_attach_penalty=lucario_attach_penalty,
                            lucario_aura_overkill_penalty=(
                                lucario_aura_overkill_penalty
                            ),
                            lucario_ordering_preference=lucario_ordering_preference,
                            dragapult_ready_attacker_penalty=(
                                dragapult_ready_attacker_penalty
                            ),
                            dragapult_evolve_penalty=dragapult_evolve_penalty,
                            dragapult_wall_penalty=dragapult_wall_penalty,
                            dragapult_budew_overstay_penalty=(
                                dragapult_budew_overstay_penalty
                            ),
                            dragapult_resource_penalty=dragapult_resource_penalty,
                            dragapult_wall_preference=dragapult_wall_preference,
                        )
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
                                "behavior_generation": behavior_generation,
                                "behavior_snapshot_id": behavior_snapshot_id,
                                "teacher_checkpoint_sha256": teacher_sha256,
                                "temperature": temperature,
                                "opponent": opponent_name,
                                "self_play": self_play,
                                "reward": adjustment.reward,
                                "tactical_reward": adjustment.reward,
                                "tactical_events": list(adjustment.events),
                                "tactical_opportunities": list(
                                    adjustment.opportunities
                                ),
                                "tactical_preferred_action": list(adjustment.preferred_action),
                                "tactical_shaping_profile": tactical_shaping_profile,
                                "tactical_shaping_revision": tactical_shaping_revision,
                                "tactical_hard_masked_options": list(
                                    hard_masked_options
                                ),
                                "tactical_search_opportunities": list(
                                    search_opportunities
                                ),
                                "tactical_option_logit_bias": list(
                                    option_logit_bias
                                ),
                                "entity_truncation_partition": (
                                    "early"
                                    if int(current.get("turn", 0) or 0) <= 4
                                    else (
                                        "mid"
                                        if int(current.get("turn", 0) or 0) <= 10
                                        else "late"
                                    )
                                ),
                                "ko": False,
                                "prize_delta": 0,
                                "final_win": False,
                                "terminal_after_action": False,
                                "evolve_target_active": (
                                    adjustment.evolve_target_active
                                ),
                                "_tactical_deferred_attack_penalty": (
                                    adjustment.deferred_attack_penalty
                                ),
                                "_tactical_deferred_preferred_action": list(
                                    adjustment.deferred_preferred_action
                                ),
                                "_tactical_deferred_attack_events": list(
                                    adjustment.deferred_attack_events
                                ),
                                "outcome": 0.0,
                            }
                        )
                        rows.append(decision)
                        recorded_decision = decision
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
                chose_attack = any(
                    0 <= index < len(options)
                    and int(options[index].get("type", -1)) == 13
                    for index in action
                )
                if recorded_decision is not None and chose_attack:
                    pending_attack_audits.pop(player, None)
                    pending_attack_audits[player] = _start_attack_audit(
                        recorded_decision, observation, player
                    )
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
    parser.add_argument(
        "--learner-deck-pool",
        type=Path,
        help="optional weighted multi-deck manifest; one learner deck is sampled per episode",
    )
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--cg-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--self-play-fraction", type=float, default=0.25)
    parser.add_argument("--learner-seat1-fraction", type=float, default=0.5)
    parser.add_argument("--archetype-weights-json")
    parser.add_argument("--agent-weights-json")
    parser.add_argument(
        "--tactical-shaping-profile",
        choices=("none", "a02", "a08", "lucario", "dragapult"),
        default="none",
    )
    parser.add_argument("--tactical-shaping-revision", type=int, default=0)
    parser.add_argument("--boss-reservation-penalty", type=float, default=0.0)
    parser.add_argument("--boss-reservation-preference", action="store_true")
    parser.add_argument("--boss-post-play-penalty", type=float, default=0.0)
    parser.add_argument("--boss-post-play-preference", action="store_true")
    parser.add_argument("--a02-poffin-decline-penalty", type=float, default=0.0)
    parser.add_argument("--a02-poffin-preference", action="store_true")
    parser.add_argument(
        "--a02-munkidori-overfill-penalty", type=float, default=0.05
    )
    parser.add_argument("--a02-bench-budget-preference", action="store_true")
    parser.add_argument("--a02-outcome-gated-ordering", action="store_true")
    parser.add_argument("--a02-projected-bench-budget", action="store_true")
    parser.add_argument("--successor-attach-preference", action="store_true")
    parser.add_argument(
        "--a08-terminal-before-evolve-mode",
        choices=("control", "end_only", "gated"),
        default="control",
    )
    parser.add_argument("--a08-gated-attack-penalty", type=float, default=0.10)
    parser.add_argument(
        "--a08-maximum-belt-support-penalty", type=float, default=0.0
    )
    parser.add_argument("--a08-maximum-belt-preference", action="store_true")
    parser.add_argument("--a08-second-attacker-reward", type=float, default=0.0)
    parser.add_argument("--a08-recovery-end-penalty", type=float, default=0.0)
    parser.add_argument("--a08-recovery-preference", action="store_true")
    parser.add_argument("--end-with-attack-penalty", type=float, default=0.0)
    parser.add_argument("--end-with-attack-preference", action="store_true")
    parser.add_argument("--lucario-evolve-penalty", type=float, default=0.0)
    parser.add_argument("--lucario-attach-penalty", type=float, default=0.0)
    # Keep the production-safe Lucario correction active for collectors spawned by
    # already-running rollout parents that predate this CLI option.  The shaping
    # branch is Lucario-only and still requires the conservative lethal/fuel/bench
    # conditions, so other profiles are unaffected.
    parser.add_argument("--lucario-aura-overkill-penalty", type=float, default=0.12)
    parser.add_argument(
        "--lucario-aura-hard-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lucario-ordering-preference", action="store_true")
    parser.add_argument("--dragapult-ready-attacker-penalty", type=float, default=0.14)
    parser.add_argument("--dragapult-evolve-penalty", type=float, default=0.16)
    parser.add_argument("--dragapult-wall-penalty", type=float, default=0.10)
    parser.add_argument("--dragapult-budew-overstay-penalty", type=float, default=0.08)
    parser.add_argument("--dragapult-resource-penalty", type=float, default=0.08)
    parser.add_argument(
        "--dragapult-wall-preference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dragapult-terminal-search-depth", type=int, choices=(0, 1, 2), default=2
    )
    parser.add_argument("--dragapult-search-bias-scale", type=float, default=1.0)
    parser.add_argument("--long-game-min-player-decisions", type=int, default=0)
    parser.add_argument("--long-game-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-decisions", type=int, default=5000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--behavior-generation", type=int, default=0)
    parser.add_argument("--behavior-snapshot-id", default="legacy")
    parser.add_argument("--role", choices=("generalist", "hard_exploiter", "diversity", "conservative"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (
        args.episodes <= 0
        or not 0 <= args.self_play_fraction <= 1
        or not 0 <= args.learner_seat1_fraction <= 1
    ):
        raise ValueError("invalid rollout episode configuration")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if args.behavior_generation < 0:
        raise ValueError("behavior generation must be non-negative")
    if args.long_game_min_player_decisions < 0:
        raise ValueError("long-game decision threshold must be non-negative")
    if not math.isfinite(args.boss_reservation_penalty) or args.boss_reservation_penalty < 0:
        raise ValueError("Boss reservation penalty must be finite and non-negative")
    extra_non_negative = {
        "Boss post-play penalty": args.boss_post_play_penalty,
        "A08 second-attacker reward": args.a08_second_attacker_reward,
        "A08 recovery end penalty": args.a08_recovery_end_penalty,
        "end-with-attack penalty": args.end_with_attack_penalty,
        "Lucario evolve penalty": args.lucario_evolve_penalty,
        "Lucario attach penalty": args.lucario_attach_penalty,
        "Lucario Aura-overkill penalty": args.lucario_aura_overkill_penalty,
        "Dragapult ready-attacker penalty": args.dragapult_ready_attacker_penalty,
        "Dragapult evolve penalty": args.dragapult_evolve_penalty,
        "Dragapult wall penalty": args.dragapult_wall_penalty,
        "Dragapult Budew-overstay penalty": args.dragapult_budew_overstay_penalty,
        "Dragapult resource penalty": args.dragapult_resource_penalty,
        "Dragapult search bias scale": args.dragapult_search_bias_scale,
    }
    for label, value in extra_non_negative.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} must be finite and non-negative")
    if (
        not math.isfinite(args.a02_poffin_decline_penalty)
        or args.a02_poffin_decline_penalty < 0
    ):
        raise ValueError("A02 Poffin decline penalty must be finite and non-negative")
    if (
        not math.isfinite(args.a02_munkidori_overfill_penalty)
        or args.a02_munkidori_overfill_penalty < 0
    ):
        raise ValueError("A02 Munkidori overfill penalty must be finite and non-negative")
    if (
        not math.isfinite(args.a08_gated_attack_penalty)
        or args.a08_gated_attack_penalty < 0
    ):
        raise ValueError("A08 gated attack penalty must be finite and non-negative")
    if (
        not math.isfinite(args.a08_maximum_belt_support_penalty)
        or args.a08_maximum_belt_support_penalty < 0
    ):
        raise ValueError(
            "A08 Maximum Belt support penalty must be finite and non-negative"
        )
    if not math.isfinite(args.long_game_weight) or args.long_game_weight < 1.0:
        raise ValueError("long-game weight must be finite and at least 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, _ = load_universal_checkpoint(args.checkpoint.resolve(), args.reference_root, device)
    teacher, _ = load_universal_checkpoint(args.teacher.resolve(), args.reference_root, device)
    model.eval()
    teacher.eval()
    runtime = load_feature_runtime(args.reference_root, args.engine_catalog)
    learner_decks = load_learner_deck_pool(
        args.learner_deck_pool.resolve() if args.learner_deck_pool else None,
        args.deck.resolve(),
    )
    pool = load_pool(args.pool.resolve())
    rng = random.Random(args.seed)
    archetype_weights = positive_weights(args.archetype_weights_json)
    agent_weights = positive_weights(args.agent_weights_json)
    seat1_count = round(args.episodes * args.learner_seat1_fraction)
    learner_seats = [1] * seat1_count + [0] * (args.episodes - seat1_count)
    rng.shuffle(learner_seats)
    behavior_sha = sha256_file(args.checkpoint)
    teacher_sha = sha256_file(args.teacher)
    temporary = args.output.with_name(args.output.name + ".tmp")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    opponents: Counter[str] = Counter()
    learner_deck_counts: Counter[str] = Counter()
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        for episode in range(args.episodes):
            learner_deck_row = rng.choices(
                learner_decks,
                weights=[float(row["weight"]) for row in learner_decks],
                k=1,
            )[0]
            self_play = rng.random() < args.self_play_fraction
            opponent = None if self_play else choose_opponent(
                pool, args.role, rng, archetype_weights, agent_weights
            )
            learner_seat = learner_seats[episode]
            rows, winner, opponent_name = play_episode(
                model=model,
                teacher=teacher,
                runtime=runtime,
                learner_deck=list(learner_deck_row["deck"]),
                opponent=opponent,
                learner_seat=learner_seat,
                cg_dir=args.cg_dir.resolve(),
                episode=episode,
                episode_id=f"{args.run_id}-{episode:06d}",
                checkpoint_sha256=behavior_sha,
                behavior_generation=args.behavior_generation,
                behavior_snapshot_id=args.behavior_snapshot_id,
                teacher_sha256=teacher_sha,
                device=device,
                temperature=args.temperature,
                max_decisions=args.max_decisions,
                tactical_shaping_profile=args.tactical_shaping_profile,
                tactical_shaping_revision=args.tactical_shaping_revision,
                boss_reservation_penalty=args.boss_reservation_penalty,
                boss_reservation_preference=args.boss_reservation_preference,
                boss_post_play_penalty=args.boss_post_play_penalty,
                boss_post_play_preference=args.boss_post_play_preference,
                a02_poffin_decline_penalty=args.a02_poffin_decline_penalty,
                a02_poffin_preference=args.a02_poffin_preference,
                a02_munkidori_overfill_penalty=(
                    args.a02_munkidori_overfill_penalty
                ),
                a02_bench_budget_preference=args.a02_bench_budget_preference,
                a02_outcome_gated_ordering=args.a02_outcome_gated_ordering,
                a02_projected_bench_budget=args.a02_projected_bench_budget,
                successor_attach_preference=args.successor_attach_preference,
                a08_terminal_before_evolve_mode=(
                    args.a08_terminal_before_evolve_mode
                ),
                a08_gated_attack_penalty=args.a08_gated_attack_penalty,
                a08_maximum_belt_support_penalty=(
                    args.a08_maximum_belt_support_penalty
                ),
                a08_maximum_belt_preference=args.a08_maximum_belt_preference,
                a08_second_attacker_reward=args.a08_second_attacker_reward,
                a08_recovery_end_penalty=args.a08_recovery_end_penalty,
                a08_recovery_preference=args.a08_recovery_preference,
                end_with_attack_penalty=args.end_with_attack_penalty,
                end_with_attack_preference=args.end_with_attack_preference,
                lucario_evolve_penalty=args.lucario_evolve_penalty,
                lucario_attach_penalty=args.lucario_attach_penalty,
                lucario_aura_overkill_penalty=args.lucario_aura_overkill_penalty,
                lucario_aura_hard_mask=args.lucario_aura_hard_mask,
                lucario_ordering_preference=args.lucario_ordering_preference,
                dragapult_ready_attacker_penalty=(
                    args.dragapult_ready_attacker_penalty
                ),
                dragapult_evolve_penalty=args.dragapult_evolve_penalty,
                dragapult_wall_penalty=args.dragapult_wall_penalty,
                dragapult_budew_overstay_penalty=(
                    args.dragapult_budew_overstay_penalty
                ),
                dragapult_resource_penalty=args.dragapult_resource_penalty,
                dragapult_wall_preference=args.dragapult_wall_preference,
                dragapult_terminal_search_depth=args.dragapult_terminal_search_depth,
                dragapult_search_bias_scale=args.dragapult_search_bias_scale,
                long_game_min_player_decisions=args.long_game_min_player_decisions,
                long_game_weight=args.long_game_weight,
            )
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                counters["tacticalRewardMilli"] += round(
                    1000.0 * float(row.get("tactical_reward", 0.0))
                )
                for event in row.get("tactical_events", []):
                    counters[f"tacticalEvent:{event}"] += 1
                for opportunity in row.get("tactical_opportunities", []):
                    counters[f"tacticalOpportunity:{opportunity}"] += 1
                if row.get("tactical_hard_masked_options"):
                    counters["tacticalHardMaskDecisions"] += 1
                    counters["tacticalHardMaskedOptions"] += len(
                        row["tactical_hard_masked_options"]
                    )
                for opportunity in row.get("tactical_search_opportunities", []):
                    counters[f"tacticalSearchOpportunity:{opportunity}"] += 1
                if any(row.get("tactical_option_logit_bias", ())):
                    counters["tacticalSearchBiasedDecisions"] += 1
                truncated = int(row.get("truncated_entities", 0) or 0)
                if truncated > 0:
                    partition = str(row.get("entity_truncation_partition", "unknown"))
                    counters["entityTruncatedDecisions"] += 1
                    counters["entityTruncatedTotal"] += truncated
                    counters[f"entityTruncatedDecision:{partition}"] += 1
                    counters[f"entityTruncatedTotal:{partition}"] += truncated
            counters["longGamePlayerTrajectories"] += sum(
                int(group_rows[0].get("long_game_episode", False))
                for group_rows in (
                    [row for row in rows if int(row["player"]) == player]
                    for player in {int(row["player"]) for row in rows}
                )
                if group_rows
            )
            counters["episodes"] += 1
            counters["decisions"] += len(rows)
            counters["selfPlayEpisodes"] += int(self_play)
            counters[f"learnerSeat{learner_seat}Episodes"] += 1
            counters["draws"] += int(winner == 2)
            if not self_play:
                counters["wins"] += int(winner == learner_seat)
                counters["losses"] += int(winner in (0, 1) and winner != learner_seat)
            opponents[opponent_name] += 1
            learner_deck_counts[str(learner_deck_row["name"])] += 1
            print(json.dumps({"progress": f"{episode + 1}/{args.episodes}", "decisions": counters["decisions"]}), flush=True)
    temporary.replace(args.output)
    error_counts: Counter[str] = Counter()
    for event, opportunity in TACTICAL_ERROR_TO_OPPORTUNITY.items():
        error_counts[opportunity] += counters.get(f"tacticalEvent:{event}", 0)
    opportunity_rates = {}
    for key, count in sorted(counters.items()):
        prefix = "tacticalOpportunity:"
        if not key.startswith(prefix):
            continue
        opportunity = key[len(prefix):]
        errors = int(error_counts.get(opportunity, 0))
        opportunity_rates[opportunity] = {
            "opportunities": int(count),
            "errors": errors,
            "errorRate": errors / count if count else None,
        }
    summary = {
        "schemaVersion": 1,
        "createdAt": utc_now(),
        "rolloutFormat": ROLLOUT_FORMAT,
        "role": args.role,
        "samplingControl": {
            "selfPlayFraction": args.self_play_fraction,
            "learnerSeat1Fraction": args.learner_seat1_fraction,
            "archetypeWeights": archetype_weights,
            "agentWeights": agent_weights,
            "tacticalShapingProfile": args.tactical_shaping_profile,
            "tacticalShapingRevision": args.tactical_shaping_revision,
            "bossReservationPenalty": args.boss_reservation_penalty,
            "bossReservationPreference": args.boss_reservation_preference,
            "bossPostPlayPenalty": args.boss_post_play_penalty,
            "bossPostPlayPreference": args.boss_post_play_preference,
            "a02PoffinDeclinePenalty": args.a02_poffin_decline_penalty,
            "a02PoffinPreference": args.a02_poffin_preference,
            "a02MunkidoriOverfillPenalty": (
                args.a02_munkidori_overfill_penalty
            ),
            "a02BenchBudgetPreference": args.a02_bench_budget_preference,
            "a02OutcomeGatedOrdering": args.a02_outcome_gated_ordering,
            "a02ProjectedBenchBudget": args.a02_projected_bench_budget,
            "successorAttachPreference": args.successor_attach_preference,
            "a08TerminalBeforeEvolveMode": args.a08_terminal_before_evolve_mode,
            "a08GatedAttackPenalty": args.a08_gated_attack_penalty,
            "a08MaximumBeltSupportPenalty": (
                args.a08_maximum_belt_support_penalty
            ),
            "a08MaximumBeltPreference": args.a08_maximum_belt_preference,
            "a08SecondAttackerReward": args.a08_second_attacker_reward,
            "a08RecoveryEndPenalty": args.a08_recovery_end_penalty,
            "a08RecoveryPreference": args.a08_recovery_preference,
            "endWithAttackPenalty": args.end_with_attack_penalty,
            "endWithAttackPreference": args.end_with_attack_preference,
            "lucarioEvolvePenalty": args.lucario_evolve_penalty,
            "lucarioAttachPenalty": args.lucario_attach_penalty,
            "lucarioAuraOverkillPenalty": args.lucario_aura_overkill_penalty,
            "lucarioAuraHardMask": args.lucario_aura_hard_mask,
            "lucarioOrderingPreference": args.lucario_ordering_preference,
            "dragapultReadyAttackerPenalty": args.dragapult_ready_attacker_penalty,
            "dragapultEvolvePenalty": args.dragapult_evolve_penalty,
            "dragapultWallPenalty": args.dragapult_wall_penalty,
            "dragapultBudewOverstayPenalty": (
                args.dragapult_budew_overstay_penalty
            ),
            "dragapultResourcePenalty": args.dragapult_resource_penalty,
            "dragapultWallPreference": args.dragapult_wall_preference,
            "dragapultTerminalSearchDepth": args.dragapult_terminal_search_depth,
            "dragapultSearchBiasScale": args.dragapult_search_bias_scale,
            "longGameMinPlayerDecisions": args.long_game_min_player_decisions,
            "longGameWeight": args.long_game_weight,
            "learnerDeckPool": str(args.learner_deck_pool.resolve()) if args.learner_deck_pool else None,
            "learnerDeckCount": len(learner_decks),
        },
        **dict(counters),
        "tacticalOpportunityRates": opportunity_rates,
        "opponents": dict(opponents),
        "learnerDecks": dict(learner_deck_counts),
        "behaviorCheckpoint": {"path": str(args.checkpoint.resolve()), "sha256": behavior_sha},
        "behaviorGeneration": args.behavior_generation,
        "behaviorSnapshotId": args.behavior_snapshot_id,
        "teacherCheckpoint": {"path": str(args.teacher.resolve()), "sha256": teacher_sha},
        "output": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output)},
        "engineSeedControlled": False,
    }
    write_json(args.output.with_suffix(args.output.suffix + ".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
