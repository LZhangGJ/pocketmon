from __future__ import annotations

import unittest
import sys
import json
from pathlib import Path


INTEGRATION = Path(__file__).resolve().parent
if str(INTEGRATION) not in sys.path:
    sys.path.insert(0, str(INTEGRATION))

from ppo_tactical_shaping import (
    TacticalShapingState,
    finalize_deferred_adjustment,
    finalize_tactical_audit_row,
    tactical_adjustment,
    tactical_hard_mask_options,
    tactical_search_logit_biases,
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
        self.assertIn(
            "a02_boss_reserved_before_other_supporter", adjustment.opportunities
        )

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

        preserved = tactical_adjustment(
            "a02", observation, options, [1], features=_Features, cards=A02_CARDS,
            a02_projected_bench_budget=True,
            a02_munkidori_overfill_penalty=0.10,
            a02_bench_budget_preference=True,
        )
        self.assertGreaterEqual(preserved.reward, 0.0)
        self.assertIn("a02_projected_bench_budget", preserved.opportunities)
        self.assertNotIn("a02_overfilled_munkidori_bench", preserved.events)

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

    def test_lucario_prefers_lethal_aura_jab_when_bench_needs_energy(self) -> None:
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {
                        "active": [{"id": 20, "hp": 340, "maxHp": 340}],
                        "bench": [{"id": 21, "hp": 70, "maxHp": 70, "energies": []}],
                        "discard": [{"id": 6}],
                    },
                    {"active": [{"id": 30, "hp": 120, "maxHp": 200}]},
                ],
            },
            "select": {},
        }
        cards = {
            6: {"name": "Basic {F} Energy", "cardType": 5, "energyType": 6},
            20: {"name": "Mega Lucario ex", "hp": 340, "megaEx": True},
            21: {"name": "Riolu", "hp": 70},
            30: {"name": "Munkidori", "hp": 200},
        }
        attacks = {
            982: {"name": "Aura Jab", "damage": 130},
            983: {"name": "Mega Brave", "damage": 270},
        }
        options = [
            {"type": 13, "attackId": 982},
            {"type": 13, "attackId": 983},
        ]
        overkill = tactical_adjustment(
            "lucario", observation, options, [1], features=_Features,
            cards=cards, attacks=attacks,
            lucario_aura_overkill_penalty=0.12,
            lucario_ordering_preference=True,
        )
        self.assertAlmostEqual(overkill.reward, -0.12)
        self.assertIn(
            "lucario_mega_brave_overkill_forfeited_aura_acceleration",
            overkill.events,
        )
        self.assertIn("lucario_aura_jab_lethal_acceleration", overkill.opportunities)
        self.assertEqual(overkill.preferred_action, (0,))

        efficient = tactical_adjustment(
            "lucario", observation, options, [0], features=_Features,
            cards=cards, attacks=attacks,
            lucario_aura_overkill_penalty=0.12,
            lucario_ordering_preference=True,
        )
        self.assertEqual(efficient.reward, 0.0)
        self.assertEqual(efficient.events, ())
        self.assertIn("lucario_aura_jab_lethal_acceleration", efficient.opportunities)

    def test_lucario_aura_preference_requires_real_acceleration_value(self) -> None:
        base = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {
                        "active": [{"id": 20}],
                        "bench": [{"id": 21, "energies": []}],
                        "discard": [],
                    },
                    {"active": [{"id": 30, "hp": 120}]},
                ],
            },
            "select": {},
        }
        cards = {
            6: {"name": "Basic {F} Energy", "cardType": 5, "energyType": 6},
        }
        attacks = {
            982: {"name": "Aura Jab", "damage": 130},
            983: {"name": "Mega Brave", "damage": 270},
        }
        options = [{"type": 13, "attackId": 982}, {"type": 13, "attackId": 983}]
        no_fuel = tactical_adjustment(
            "lucario", base, options, [1], features=_Features,
            cards=cards, attacks=attacks,
            lucario_aura_overkill_penalty=0.12,
            lucario_ordering_preference=True,
        )
        self.assertEqual(no_fuel.reward, 0.0)
        self.assertNotIn("lucario_aura_jab_lethal_acceleration", no_fuel.opportunities)

        high_hp = json.loads(json.dumps(base))
        high_hp["current"]["players"][0]["discard"] = [{"id": 6}]
        high_hp["current"]["players"][1]["active"][0]["hp"] = 200
        needs_damage = tactical_adjustment(
            "lucario", high_hp, options, [1], features=_Features,
            cards=cards, attacks=attacks,
            lucario_aura_overkill_penalty=0.12,
            lucario_ordering_preference=True,
        )
        self.assertEqual(needs_damage.reward, 0.0)
        self.assertNotIn("lucario_aura_jab_lethal_acceleration", needs_damage.opportunities)

    def test_lucario_hard_mask_removes_only_dominated_mega_brave(self) -> None:
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {
                        "active": [{"id": 20}],
                        "bench": [{"id": 21, "energies": []}],
                        "discard": [{"id": 6}],
                    },
                    {"active": [{"id": 30, "hp": 120}]},
                ],
            },
            "select": {},
        }
        cards = {6: {"cardType": 5, "energyType": 6}}
        attacks = {
            982: {"name": "Aura Jab", "damage": 130},
            983: {"name": "Mega Brave", "damage": 270},
        }
        options = [{"type": 13, "attackId": 982}, {"type": 13, "attackId": 983}]
        self.assertEqual(
            tactical_hard_mask_options(
                "lucario", observation, options, cards=cards, attacks=attacks,
                lucario_aura_hard_mask=True,
            ),
            (1,),
        )
        observation["current"]["players"][1]["active"][0]["hp"] = 200
        self.assertEqual(
            tactical_hard_mask_options(
                "lucario", observation, options, cards=cards, attacks=attacks,
                lucario_aura_hard_mask=True,
            ),
            (),
        )

    def test_terminal_ko_masks_every_nonwinning_action(self) -> None:
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [
                    {"active": [{"id": 20}], "prize": [1, 2, 3, 4, 5]},
                    {"active": [{"id": 30, "hp": 120}]},
                ],
            }
        }
        options = [
            {"type": 13, "attackId": 982},
            {"type": 13, "attackId": 983},
            {"type": 14},
        ]
        attacks = {
            982: {"name": "Aura Jab", "damage": 130},
            983: {"name": "Weak Hit", "damage": 20},
        }
        self.assertEqual(
            tactical_hard_mask_options(
                "lucario", observation, options, cards={}, attacks=attacks
            ),
            (1, 2),
        )

    def test_forced_deckout_masks_fixed_draw_but_not_recycle(self) -> None:
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [{"active": [], "deckCount": 2, "prize": []}, {}],
            }
        }
        cards = {
            50: {"name": "Risky Draw", "skills": [{"text": "Draw 3 cards."}]},
            51: {"name": "Recycle", "skills": [{"text": "Put your hand on the bottom of your deck. Draw 6 cards."}]},
        }
        options = [
            {"type": 7, "card": {"id": 50}},
            {"type": 7, "card": {"id": 51}},
            {"type": 14},
        ]
        self.assertEqual(
            tactical_hard_mask_options(
                "dragapult", observation, options, features=_Features,
                cards=cards, attacks={}
            ),
            (0,),
        )

    def test_dragapult_legal_ex_evolution_masks_end_only(self) -> None:
        observation = {
            "current": {
                "yourIndex": 0,
                "players": [{"active": [{"id": 20}], "prize": []}, {}],
            }
        }
        cards = {40: {"name": "Dragapult ex"}}
        options = [
            {"type": 9, "card": {"id": 40}},
            {"type": 8, "card": {"id": 41}},
            {"type": 14},
        ]
        self.assertEqual(
            tactical_hard_mask_options(
                "dragapult", observation, options, features=_Features,
                cards=cards, attacks={}
            ),
            (2,),
        )

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


class DragapultTempoShapingTest(unittest.TestCase):
    cards = {
        121: {"name": "Dragapult ex", "ex": True},
        235: {"name": "Budew"},
        112: {"name": "Munkidori"},
        345: {
            "name": "Crustle",
            "skills": [{
                "text": "Prevent all damage done to this Pokémon by attacks from your opponent's Pokémon ex."
            }],
        },
        533: {"name": "Crustle", "skills": [{"text": "Sturdy"}]},
        1152: {"name": "Poké Pad", "cardType": 2},
    }
    attacks = {
        154: {"name": "Phantom Dive", "damage": 200, "energies": [2, 5]},
        323: {"name": "Itchy Pollen", "damage": 10},
        141: {"name": "Mind Bend", "damage": 60},
    }

    def observation(self, *, opponent=345, deck=20, hand=4):
        return {
            "current": {
                "yourIndex": 0,
                "turn": 7,
                "firstPlayer": 1,
                "players": [
                    {
                        "active": [{"id": 235, "serial": 1}],
                        "bench": [{"id": 121, "serial": 2, "energies": [2, 5]}],
                        "deckCount": deck,
                        "handCount": hand,
                    },
                    {
                        "active": [{"id": opponent, "hp": 200}],
                        "bench": [{"id": 112, "hp": 120}],
                    },
                ],
            },
            "select": {},
        }

    def test_phantom_dive_into_crustle_wall_is_never_penalized(self):
        observation = self.observation()
        observation["current"]["players"][0] = {
            "active": [{"id": 121, "energies": [2, 5]}], "bench": []
        }
        candidate = tactical_adjustment(
            "dragapult", observation, [{"type": 13, "attackId": 154}], [0],
            features=_Features, cards=self.cards, attacks=self.attacks,
            dragapult_wall_penalty=0.10,
        )
        self.assertEqual(candidate.reward, 0.0)
        self.assertEqual(candidate.deferred_attack_penalty, 0.0)
        self.assertEqual(candidate.events, ())
        self.assertIn("dragapult_phantom_full_value_search", candidate.opportunities)

    def test_ready_dragapult_takeover_penalty_is_outcome_gated(self):
        candidate = tactical_adjustment(
            "dragapult", self.observation(), [{"type": 13, "attackId": 323}], [0],
            features=_Features, cards=self.cards, attacks=self.attacks,
            dragapult_ready_attacker_penalty=0.14,
        )
        self.assertAlmostEqual(candidate.deferred_attack_penalty, 0.14)
        missed = finalize_deferred_adjustment(
            candidate, ko=False, prize_delta=0, terminal_after_action=False
        )
        self.assertAlmostEqual(missed.reward, -0.14)
        self.assertIn("dragapult_delayed_ready_attacker_with_stall_action", missed.events)
        scored = finalize_deferred_adjustment(
            candidate, ko=True, prize_delta=1, terminal_after_action=False
        )
        self.assertEqual(scored.reward, 0.0)

    def test_end_before_dragapult_evolution_prefers_evolution(self):
        options = [
            {"type": 9, "card": {"id": 121}, "target": {"id": 112}},
            {"type": 14},
        ]
        candidate = tactical_adjustment(
            "dragapult", self.observation(), options, [1], features=_Features,
            cards=self.cards, attacks=self.attacks,
            dragapult_evolve_penalty=0.16,
        )
        self.assertAlmostEqual(candidate.reward, -0.30)
        self.assertIn("dragapult_terminal_before_safe_dragapult_evolution", candidate.events)
        self.assertEqual(candidate.preferred_action, (0,))

    def test_thin_deck_poke_pad_is_shaped_but_sturdy_is_not_ex_wall(self):
        candidate = tactical_adjustment(
            "dragapult", self.observation(opponent=533, deck=3, hand=12),
            [{"type": 7, "card": {"id": 1152}}], [0], features=_Features,
            cards=self.cards, attacks=self.attacks,
        )
        self.assertAlmostEqual(candidate.reward, -0.08)
        self.assertIn("dragapult_searched_thin_deck_into_wall", candidate.events)
        self.assertNotIn("dragapult_phantom_full_value_search", candidate.opportunities)

    def test_two_ply_search_prefers_ready_evolution_and_preserves_phantom(self):
        observation = self.observation(deck=3, hand=12)
        options = [
            {"type": 9, "card": {"id": 121}, "target": {"id": 112, "energies": [2, 5]}},
            {"type": 13, "attackId": 323},
            {"type": 13, "attackId": 154},
            {"type": 7, "card": {"id": 1152}},
            {"type": 14},
        ]
        biases, opportunities = tactical_search_logit_biases(
            "dragapult", observation, options, features=_Features,
            cards=self.cards, attacks=self.attacks, depth=2, scale=1.0,
        )
        self.assertGreater(biases[0], 1.0)
        self.assertLess(biases[1], 0.0)
        self.assertGreater(biases[2], 0.0)
        self.assertLess(biases[3], 0.0)
        self.assertLess(biases[4], 0.0)
        self.assertIn("dragapult_search_next_attacker", opportunities)


if __name__ == "__main__":
    unittest.main()
