from __future__ import annotations

import unittest
import sys
from pathlib import Path


INTEGRATION = Path(__file__).resolve().parent
if str(INTEGRATION) not in sys.path:
    sys.path.insert(0, str(INTEGRATION))

from ppo_tactical_shaping import (
    TacticalShapingState,
    finalize_deferred_adjustment,
    finalize_tactical_audit_row,
    tactical_adjustment,
)


class _Features:
    @staticmethod
    def resolve_option_cards(observation, option):
        return option.get("card"), option.get("target")


class A08TacticalShapingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = {
            "current": {
                "yourIndex": 0,
                "players": [{"active": [], "bench": []}, {}],
            },
            "select": {},
        }
        self.options = [
            {"type": 9, "inPlayArea": 4, "card": {"id": 90}},
            {"type": 13},
            {"type": 14},
        ]
        self.cards = {90: {"name": "Thwackey", "hp": 100, "skills": [{}]}}

    def adjustment(self, action: int, mode: str):
        return tactical_adjustment(
            "a08",
            self.observation,
            self.options,
            [action],
            features=_Features,
            cards=self.cards,
            a08_terminal_before_evolve_mode=mode,
            a08_gated_attack_penalty=0.10,
        )

    def test_control_preserves_legacy_attack_and_end_penalty(self) -> None:
        for action in (1, 2):
            with self.subTest(action=action):
                adjustment = self.adjustment(action, "control")
                self.assertEqual(adjustment.reward, -0.35)
                self.assertEqual(
                    adjustment.events, ("a08_terminal_before_evolve",)
                )
                self.assertEqual(adjustment.preferred_action, (0,))

    def test_end_only_penalizes_end_but_not_attack(self) -> None:
        attack = self.adjustment(1, "end_only")
        self.assertEqual(attack.reward, 0.0)
        self.assertEqual(attack.events, ())
        self.assertEqual(attack.preferred_action, ())

        end = self.adjustment(2, "end_only")
        self.assertEqual(end.reward, -0.35)
        self.assertEqual(end.events, ("a08_terminal_before_evolve",))
        self.assertEqual(end.preferred_action, (0,))

    def test_gated_non_scoring_nonterminal_attack_gets_small_penalty(self) -> None:
        candidate = self.adjustment(1, "gated")
        self.assertEqual(candidate.reward, 0.0)
        self.assertEqual(candidate.preferred_action, ())
        self.assertEqual(candidate.deferred_attack_penalty, 0.10)
        self.assertTrue(candidate.evolve_target_active)

        resolved = finalize_deferred_adjustment(
            candidate, ko=False, prize_delta=0, terminal_after_action=False
        )
        self.assertEqual(resolved.reward, -0.10)
        self.assertEqual(
            resolved.events,
            ("a08_nonlethal_attack_before_high_value_evolve",),
        )
        self.assertEqual(resolved.preferred_action, (0,))

    def test_gated_ko_prize_and_terminal_attacks_are_exempt(self) -> None:
        candidate = self.adjustment(1, "gated")
        outcomes = (
            {"ko": True, "prize_delta": 0, "terminal_after_action": False},
            {"ko": False, "prize_delta": 1, "terminal_after_action": False},
            {"ko": False, "prize_delta": 0, "terminal_after_action": True},
        )
        for outcome in outcomes:
            with self.subTest(outcome=outcome):
                resolved = finalize_deferred_adjustment(candidate, **outcome)
                self.assertEqual(resolved.reward, 0.0)
                self.assertEqual(resolved.events, ())
                self.assertEqual(resolved.preferred_action, ())

    def test_gated_attack_requires_high_value_evolution(self) -> None:
        options = [
            {
                "type": 9,
                "inPlayArea": 5,
                "card": {"id": 347},
                "target": {"id": 346, "maxHp": 70},
            },
            {"type": 13},
        ]
        cards = {347: {"name": "Dipplin", "hp": 80, "skills": []}}
        adjustment = tactical_adjustment(
            "a08",
            self.observation,
            options,
            [1],
            features=_Features,
            cards=cards,
            a08_terminal_before_evolve_mode="gated",
        )
        self.assertEqual(adjustment.reward, 0.0)
        self.assertEqual(adjustment.deferred_attack_penalty, 0.0)
        self.assertEqual(adjustment.events, ())

    def test_audit_row_exposes_outcome_fields_and_removes_internal_state(self) -> None:
        row = {
            "tactical_reward": 0.0,
            "tactical_events": [],
            "tactical_preferred_action": [],
            "ko": False,
            "prize_delta": 0,
            "terminal_after_action": False,
            "evolve_target_active": True,
            "_tactical_deferred_attack_penalty": 0.10,
            "_tactical_deferred_preferred_action": [0],
        }
        finalize_tactical_audit_row(row, final_win=True)
        self.assertTrue(row["final_win"])
        self.assertFalse(row["ko"])
        self.assertEqual(row["prize_delta"], 0)
        self.assertTrue(row["evolve_target_active"])
        self.assertEqual(row["tactical_reward"], -0.10)
        self.assertNotIn("_tactical_deferred_attack_penalty", row)
        self.assertNotIn("_tactical_deferred_preferred_action", row)

    def test_maximum_belt_prefers_attacker_over_support(self) -> None:
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {
                        "active": [{"id": 90, "serial": 10}],
                        "bench": [
                            {"id": 346, "serial": 11},
                            {"id": 347, "serial": 12},
                        ],
                    },
                    {},
                ],
            },
            "select": {},
        }
        options = [
            {"type": 8, "card": {"id": 200}, "target": {"id": 90}},
            {"type": 8, "card": {"id": 200}, "target": {"id": 346}},
            {"type": 8, "card": {"id": 200}, "target": {"id": 347}},
        ]
        cards = {
            90: {"name": "Thwackey"},
            200: {"name": "Maximum Belt"},
            346: {"name": "Applin"},
            347: {"name": "Dipplin"},
        }
        adjustment = tactical_adjustment(
            "a08",
            observation,
            options,
            [0],
            features=_Features,
            cards=cards,
            a08_terminal_before_evolve_mode="gated",
            a08_maximum_belt_support_penalty=0.06,
            a08_maximum_belt_preference=True,
        )
        self.assertAlmostEqual(adjustment.reward, -0.06)
        self.assertEqual(
            adjustment.events,
            ("a08_maximum_belt_on_support_with_attacker_available",),
        )
        self.assertEqual(adjustment.preferred_action, (2,))

        attacker = tactical_adjustment(
            "a08",
            observation,
            options,
            [2],
            features=_Features,
            cards=cards,
            a08_terminal_before_evolve_mode="gated",
            a08_maximum_belt_support_penalty=0.06,
            a08_maximum_belt_preference=True,
        )
        self.assertAlmostEqual(attacker.reward, 0.0)
        self.assertEqual(attacker.events, ())


A02_CARDS = {
    1: {"name": "Marnie's Grimmsnarl ex", "cardType": 0},
    2: {"name": "Munkidori", "cardType": 0},
    3: {"name": "Marnie's Impidimp", "cardType": 0},
    4: {"name": "Snorunt", "cardType": 0},
    5: {"name": "Mega Lopunny ex", "cardType": 0},
    6: {"name": "Dunsparce", "cardType": 0},
    7: {"name": "Froslass", "cardType": 0},
    1086: {"name": "Buddy-Buddy Poffin", "cardType": 2},
    1182: {"name": "Boss’s Orders", "cardType": 3},
    1231: {"name": "Dawn", "cardType": 3},
}
A02_OPTIONS = [
    {"type": 7, "card": {"id": 1182}},
    {"type": 7, "card": {"id": 1231}},
    {"type": 13, "attackId": 501},
]


def a02_observation(*, supporter_played: bool = False) -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "turn": 9,
            "energyAttached": True,
            "supporterPlayed": supporter_played,
            "players": [
                {"active": [{"id": 1, "hp": 320}], "bench": []},
                {
                    "active": [{"id": 5, "hp": 330}],
                    "bench": [{"id": 6, "hp": 70}],
                },
            ],
        },
        "select": {},
    }


class A02BossReservationTest(unittest.TestCase):
    def adjustment(self, action, *, damage=180, state=None, options=A02_OPTIONS):
        return tactical_adjustment(
            "a02",
            a02_observation(),
            options,
            action,
            features=_Features,
            cards=A02_CARDS,
            attacks={501: {"damage": damage}},
            state=state,
            boss_reservation_penalty=0.09,
            boss_reservation_preference=True,
        )

    def test_dawn_before_attack_consumes_reserved_boss(self) -> None:
        state = TacticalShapingState()
        dawn = self.adjustment([1], state=state)
        self.assertEqual(
            dawn.events,
            ("consumed_supporter_before_boss_low_hp_bench",),
        )
        self.assertAlmostEqual(dawn.reward, -0.09)
        self.assertEqual(dawn.preferred_action, (0,))

        attack = tactical_adjustment(
            "a02",
            a02_observation(supporter_played=True),
            [A02_OPTIONS[2]],
            [0],
            features=_Features,
            cards=A02_CARDS,
            attacks={501: {"damage": 180}},
            state=state,
            boss_reservation_penalty=0.09,
            boss_reservation_preference=True,
        )
        self.assertNotIn(
            "consumed_supporter_before_boss_low_hp_bench", attack.events
        )

    def test_low_hp_bench_not_koable_is_not_penalized(self) -> None:
        adjustment = self.adjustment(
            [1], damage=60, state=TacticalShapingState()
        )
        self.assertNotIn(
            "consumed_supporter_before_boss_low_hp_bench", adjustment.events
        )
        self.assertAlmostEqual(adjustment.reward, 0.0)
        self.assertEqual(adjustment.preferred_action, ())

    def test_boss_first_is_not_penalized(self) -> None:
        adjustment = self.adjustment([0], state=TacticalShapingState())
        self.assertNotIn(
            "consumed_supporter_before_boss_low_hp_bench", adjustment.events
        )
        self.assertAlmostEqual(adjustment.reward, 0.0)
        self.assertEqual(adjustment.preferred_action, ())

    def test_existing_row_local_boss_event_is_compatible(self) -> None:
        adjustment = tactical_adjustment(
            "a02",
            a02_observation(),
            [A02_OPTIONS[0], A02_OPTIONS[2]],
            [1],
            features=_Features,
            cards=A02_CARDS,
        )
        self.assertIn("a02_terminal_before_boss_low_hp_bench", adjustment.events)
        self.assertAlmostEqual(adjustment.reward, -0.12)
        self.assertEqual(adjustment.preferred_action, (0,))


class A02SetupShapingTest(unittest.TestCase):
    def test_poffin_decline_prefers_snorunt_when_froslass_is_in_hand(self) -> None:
        observation = a02_observation()
        observation["select"] = {"effect": {"id": 1086}}
        observation["current"]["players"][0]["hand"] = [{"id": 7}]
        options = [{"type": 3, "card": {"id": 4}}]
        declined = tactical_adjustment(
            "a02",
            observation,
            options,
            [],
            features=_Features,
            cards=A02_CARDS,
            a02_poffin_decline_penalty=0.08,
            a02_poffin_preference=True,
        )
        self.assertAlmostEqual(declined.reward, -0.08)
        self.assertEqual(
            declined.events,
            ("a02_declined_snorunt_after_poffin_with_froslass_in_hand",),
        )
        self.assertEqual(declined.preferred_action, (0,))

        selected = tactical_adjustment(
            "a02",
            observation,
            options,
            [0],
            features=_Features,
            cards=A02_CARDS,
            a02_poffin_decline_penalty=0.08,
            a02_poffin_preference=True,
        )
        self.assertAlmostEqual(selected.reward, 0.04)
        self.assertNotIn(
            "a02_declined_snorunt_after_poffin_with_froslass_in_hand",
            selected.events,
        )
        self.assertIn("a02_started_froslass_setup", selected.events)

    def test_poffin_decline_without_froslass_in_hand_is_not_shaped(self) -> None:
        observation = a02_observation()
        observation["select"] = {"effect": {"id": 1086}}
        observation["current"]["players"][0]["hand"] = []
        adjustment = tactical_adjustment(
            "a02",
            observation,
            [{"type": 3, "card": {"id": 4}}],
            [],
            features=_Features,
            cards=A02_CARDS,
            a02_poffin_decline_penalty=0.08,
            a02_poffin_preference=True,
        )
        self.assertAlmostEqual(adjustment.reward, 0.0)
        self.assertEqual(adjustment.events, ())

    def test_munkidori_last_slot_prefers_missing_engine_piece(self) -> None:
        observation = a02_observation()
        observation["current"]["players"][0] = {
            "active": [{"id": 1}],
            "bench": [
                {"id": 2},
                {"id": 3},
                {"id": 3},
                {"id": 4},
            ],
            "hand": [],
        }
        options = [
            {"type": 7, "card": {"id": 2}},
            {"type": 9, "card": {"id": 7}, "target": {"id": 4}},
        ]
        adjustment = tactical_adjustment(
            "a02",
            observation,
            options,
            [0],
            features=_Features,
            cards=A02_CARDS,
            a02_munkidori_overfill_penalty=0.10,
            a02_bench_budget_preference=True,
        )
        self.assertAlmostEqual(adjustment.reward, -0.10)
        self.assertEqual(adjustment.events, ("a02_overfilled_munkidori_bench",))
        self.assertEqual(adjustment.preferred_action, (1,))


class Revision7OutcomeAwareShapingTest(unittest.TestCase):
    def test_a02_nonlethal_ordering_is_deferred_and_ko_is_exempt(self) -> None:
        observation = a02_observation()
        observation["current"]["players"][0] = {
            "active": [{"id": 1, "hp": 320}],
            "bench": [{"id": 8, "hp": 100}],
            "hand": [],
        }
        cards = {**A02_CARDS, 8: {"name": "Marnie's Morgrem", "cardType": 0}}
        options = [
            {"type": 9, "card": {"id": 1}, "target": {"id": 8}},
            {"type": 13, "attackId": 501},
        ]
        candidate = tactical_adjustment(
            "a02",
            observation,
            options,
            [1],
            features=_Features,
            cards=cards,
            attacks={501: {"damage": 180}},
            a02_outcome_gated_ordering=True,
        )
        self.assertEqual(candidate.reward, 0.0)
        self.assertAlmostEqual(candidate.deferred_attack_penalty, 0.30)
        self.assertIn("a02_second_grimmsnarl_before_terminal", candidate.opportunities)
        nonlethal = finalize_deferred_adjustment(
            candidate, ko=False, prize_delta=0, terminal_after_action=False
        )
        self.assertAlmostEqual(nonlethal.reward, -0.30)
        self.assertIn(
            "a02_nonlethal_attack_before_second_grimmsnarl", nonlethal.events
        )
        scored = finalize_deferred_adjustment(
            candidate, ko=True, prize_delta=1, terminal_after_action=False
        )
        self.assertEqual(scored.reward, 0.0)
        self.assertEqual(scored.events, ())

    def test_boss_post_play_end_is_shaped_but_attack_is_not(self) -> None:
        state = TacticalShapingState()
        tactical_adjustment(
            "a02",
            a02_observation(),
            A02_OPTIONS,
            [0],
            features=_Features,
            cards=A02_CARDS,
            attacks={501: {"damage": 180}},
            state=state,
        )
        follow_options = [A02_OPTIONS[2], {"type": 14}]
        ended = tactical_adjustment(
            "a02",
            a02_observation(supporter_played=True),
            follow_options,
            [1],
            features=_Features,
            cards=A02_CARDS,
            attacks={501: {"damage": 180}},
            state=state,
            boss_post_play_penalty=0.08,
            boss_post_play_preference=True,
        )
        self.assertIn("a02_ended_after_boss_with_attack_available", ended.events)
        self.assertEqual(ended.preferred_action, (0,))

        state = TacticalShapingState()
        tactical_adjustment(
            "a02", a02_observation(), A02_OPTIONS, [0],
            features=_Features, cards=A02_CARDS,
            attacks={501: {"damage": 180}}, state=state,
        )
        attacked = tactical_adjustment(
            "a02", a02_observation(supporter_played=True), follow_options, [0],
            features=_Features, cards=A02_CARDS,
            attacks={501: {"damage": 180}}, state=state,
            boss_post_play_penalty=0.08,
        )
        self.assertNotIn("a02_ended_after_boss_with_attack_available", attacked.events)

    def test_projected_third_munkidori_reserves_engine_slots(self) -> None:
        observation = a02_observation()
        observation["current"]["players"][0] = {
            "active": [{"id": 1}],
            "bench": [{"id": 2}, {"id": 2}, {"id": 3}],
            "hand": [],
        }
        options = [
            {"type": 7, "card": {"id": 2}},
            {"type": 7, "card": {"id": 4}},
        ]
        adjustment = tactical_adjustment(
            "a02", observation, options, [0], features=_Features, cards=A02_CARDS,
            a02_projected_bench_budget=True,
            a02_munkidori_overfill_penalty=0.10,
            a02_bench_budget_preference=True,
        )
        self.assertAlmostEqual(adjustment.reward, -0.10)
        self.assertIn("a02_projected_bench_budget", adjustment.opportunities)
        self.assertEqual(adjustment.preferred_action, (1,))

    def test_successor_attach_prefers_bench_morgrem(self) -> None:
        observation = a02_observation()
        observation["current"]["energyAttached"] = False
        observation["current"]["players"][0] = {
            "active": [{"id": 1, "serial": 10, "hp": 90, "maxHp": 320}],
            "bench": [{"id": 8, "serial": 11, "hp": 100, "maxHp": 100}],
            "hand": [],
        }
        cards = {**A02_CARDS, 8: {"name": "Marnie's Morgrem", "cardType": 0}}
        options = [
            {"type": 8, "target": {"id": 1, "serial": 10, "hp": 90, "maxHp": 320}},
            {"type": 8, "target": {"id": 8, "serial": 11, "hp": 100, "maxHp": 100}},
            {"type": 13, "attackId": 501},
        ]
        adjustment = tactical_adjustment(
            "a02", observation, options, [2], features=_Features, cards=cards,
            attacks={501: {"damage": 180}},
            a02_outcome_gated_ordering=True,
            successor_attach_preference=True,
        )
        self.assertEqual(adjustment.deferred_preferred_action, (1,))

    def test_lucario_attack_penalties_are_outcome_gated(self) -> None:
        observation = {
            "current": {
                "yourIndex": 0,
                "energyAttached": False,
                "players": [
                    {"active": [{"id": 20, "serial": 20}], "bench": [{"id": 21, "serial": 21}]},
                    {},
                ],
            },
            "select": {},
        }
        cards = {
            20: {"name": "Mega Lucario ex", "hp": 340, "megaEx": True},
            21: {"name": "Riolu", "hp": 70},
        }
        options = [
            {"type": 9, "card": {"id": 20}, "target": {"id": 21}},
            {"type": 8, "target": {"id": 21, "serial": 21}},
            {"type": 13, "attackId": 600},
        ]
        candidate = tactical_adjustment(
            "lucario", observation, options, [2], features=_Features, cards=cards,
            attacks={600: {"damage": 130}},
            lucario_evolve_penalty=0.12,
            lucario_attach_penalty=0.06,
            lucario_ordering_preference=True,
            successor_attach_preference=True,
        )
        self.assertAlmostEqual(candidate.deferred_attack_penalty, 0.18)
        resolved = finalize_deferred_adjustment(
            candidate, ko=False, prize_delta=0, terminal_after_action=False
        )
        self.assertAlmostEqual(resolved.reward, -0.18)
        self.assertEqual(len(resolved.events), 2)
        exempt = finalize_deferred_adjustment(
            candidate, ko=True, prize_delta=0, terminal_after_action=False
        )
        self.assertEqual(exempt.reward, 0.0)

    def test_a08_rewards_second_attacker_and_penalizes_recovery_end(self) -> None:
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [{"active": [{"id": 347}], "bench": []}, {}],
            },
            "select": {},
        }
        cards = {346: {"name": "Applin"}, 347: {"name": "Dipplin"}}
        options = [{"type": 7, "card": {"id": 346}}, {"type": 14}]
        setup = tactical_adjustment(
            "a08", observation, options, [0], features=_Features, cards=cards,
            a08_terminal_before_evolve_mode="gated",
            a08_second_attacker_reward=0.04,
        )
        self.assertAlmostEqual(setup.reward, 0.04)
        ended = tactical_adjustment(
            "a08", observation, options, [1], features=_Features, cards=cards,
            a08_terminal_before_evolve_mode="gated",
            a08_recovery_end_penalty=0.10,
            a08_recovery_preference=True,
        )
        self.assertAlmostEqual(ended.reward, -0.10)
        self.assertEqual(ended.preferred_action, (0,))


if __name__ == "__main__":
    unittest.main()
