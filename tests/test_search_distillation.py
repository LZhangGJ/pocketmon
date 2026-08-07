from __future__ import annotations

import unittest

from scripts.train_search_distilled_actor import build_teacher_rows


def row(episode: str, step: int, option: int, target: float, std: float = 0.0) -> dict:
    return {
        "episode_id": episode,
        "observation_step": step,
        "player": 0,
        "option_index": option,
        "q_target": target,
        "q_target_std": std,
        "action": [option],
    }


class SearchDistillationTests(unittest.TestCase):
    def test_selects_confident_best_counterfactual_action(self) -> None:
        teachers, audit = build_teacher_rows(
            [row("e", 1, 0, 0.1), row("e", 1, 1, 0.7, 0.2)],
            min_margin=0.15,
            max_target_std=0.5,
        )
        self.assertEqual(len(teachers), 1)
        self.assertEqual(teachers[0]["action"], [1])
        self.assertAlmostEqual(teachers[0]["teacher_margin"], 0.6)
        self.assertEqual(audit["accepted"], 1)

    def test_rejects_low_margin_and_high_uncertainty(self) -> None:
        teachers, audit = build_teacher_rows(
            [
                row("low", 1, 0, 0.40), row("low", 1, 1, 0.45),
                row("uncertain", 1, 0, 0.0), row("uncertain", 1, 1, 0.8, 0.9),
            ],
            min_margin=0.15,
            max_target_std=0.5,
        )
        self.assertEqual(teachers, [])
        self.assertEqual(audit["low_margin"], 1)
        self.assertEqual(audit["high_uncertainty"], 1)


if __name__ == "__main__":
    unittest.main()
