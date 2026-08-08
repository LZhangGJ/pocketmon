from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "outputs" / "top_ladder_2026_08_07"


def main() -> None:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    quality = pd.read_csv(OUT / "data_quality.csv")
    archetypes = pd.read_csv(OUT / "archetype_summary.csv")
    comparison = pd.read_csv(OUT / "day_comparison.csv")
    appearances = pd.read_csv(OUT / "elite_deck_appearances.csv")
    cards = pd.read_csv(OUT / "card_usage.csv")
    matchups = pd.read_csv(OUT / "matchups_long.csv")
    decklists = pd.read_csv(OUT / "representative_decklists.csv")

    checks: dict[str, dict[str, object]] = {}

    def record(name: str, passed: bool, evidence: object) -> None:
        checks[name] = {"passed": bool(passed), "evidence": evidence}

    current = appearances[appearances["cohort"] == "current_elite"]
    previous = appearances[appearances["cohort"] == "previous_elite"]
    record(
        "one_deck_row_per_episode_player",
        not appearances.duplicated(["cohort", "episode_id", "player_index"]).any(),
        {"rows": int(len(appearances)), "duplicate_rows": int(appearances.duplicated(["cohort", "episode_id", "player_index"]).sum())},
    )
    record(
        "elite_appearance_count_matches_games",
        len(current) == 2 * int(summary["elite_valid_games"]),
        {"appearances": int(len(current)), "valid_games_x2": 2 * int(summary["elite_valid_games"])},
    )
    record(
        "archetype_appearances_sum",
        int(archetypes["appearances"].sum()) == len(current),
        {"archetype_sum": int(archetypes["appearances"].sum()), "appearance_rows": int(len(current))},
    )
    record(
        "current_usage_shares_sum_to_one",
        abs(float(archetypes["usage_share"].sum()) - 1.0) < 1e-9,
        float(archetypes["usage_share"].sum()),
    )
    record(
        "comparison_shares_sum_to_one",
        abs(float(comparison["current_usage_share"].sum()) - 1.0) < 1e-9
        and abs(float(comparison["previous_usage_share"].sum()) - 1.0) < 1e-9,
        {
            "current": float(comparison["current_usage_share"].sum()),
            "previous": float(comparison["previous_usage_share"].sum()),
            "previous_rows": int(len(previous)),
        },
    )

    weighted_win_rate = float(
        (archetypes["appearances"] * archetypes["all_win_rate"]).sum()
        / archetypes["appearances"].sum()
    )
    record(
        "two_player_rewards_conserve_wins",
        abs(weighted_win_rate - 0.5) < 1e-9,
        weighted_win_rate,
    )

    deck_sums = decklists.groupby(["archetype_id", "representative_exact_deck_id"])["count"].sum()
    record(
        "representative_decks_have_60_cards",
        bool((deck_sums == 60).all()),
        {"min": int(deck_sums.min()), "max": int(deck_sums.max()), "decks": int(len(deck_sums))},
    )
    record(
        "card_usage_has_official_mapping",
        not (cards["card_group"] == "Unknown").any() and cards["card_name_en"].notna().all(),
        {"unknown_group_rows": int((cards["card_group"] == "Unknown").sum()), "missing_names": int(cards["card_name_en"].isna().sum())},
    )
    record(
        "card_inclusion_rates_are_valid",
        bool(cards["elite_inclusion_rate"].between(0, 1).all())
        and bool(cards["field_inclusion_rate"].between(0, 1).all()),
        {
            "elite_min": float(cards["elite_inclusion_rate"].min()),
            "elite_max": float(cards["elite_inclusion_rate"].max()),
            "field_min": float(cards["field_inclusion_rate"].min()),
            "field_max": float(cards["field_inclusion_rate"].max()),
        },
    )

    reciprocal = matchups.merge(
        matchups,
        left_on=["archetype_id", "opponent_archetype_id"],
        right_on=["opponent_archetype_id", "archetype_id"],
        suffixes=("_left", "_right"),
    )
    nonmirror = reciprocal[reciprocal["archetype_id_left"] != reciprocal["opponent_archetype_id_left"]]
    n_mismatch = int((nonmirror["matchup_appearances_left"] != nonmirror["matchup_appearances_right"]).sum())
    max_win_sum_error = float(
        np.abs(nonmirror["matchup_win_rate_left"] + nonmirror["matchup_win_rate_right"] - 1).max()
    ) if len(nonmirror) else 0.0
    record(
        "matchups_are_reciprocal",
        n_mismatch == 0 and max_win_sum_error < 1e-9,
        {"sample_size_mismatches": n_mismatch, "max_win_rate_sum_error": max_win_sum_error},
    )

    current_quality = quality.loc[quality["date"] == summary["as_of_date"]].iloc[0]
    missing_files = int(current_quality["manifest_games"] - current_quality["files_present_for_manifest"])
    invalid_loaded_games = int(current_quality["files_present_for_manifest"] - current_quality["valid_games"])
    all_passed = all(item["passed"] for item in checks.values())
    coverage_complete = float(current_quality["valid_game_coverage"]) == 1.0
    assessment = "Ready to share" if all_passed and coverage_complete else "Share with caveats" if all_passed else "Needs revision"
    report = {
        "overall_assessment": assessment,
        "checks": checks,
        "required_caveats": [
            f"Current manifest has {missing_files} replay files unavailable from the official daily dataset.",
            f"A further {invalid_loaded_games} loaded replays lack a valid two-player outcome and are excluded.",
            "Player-level scores cannot be mapped to agent identities; min_score is used to guarantee both players clear the cohort cutoff.",
            "Wilson intervals treat appearances as independent even though teams play repeatedly; small single-team archetypes are directional only.",
            "Manifest timestamps do not declare a timezone offset.",
        ],
    }
    (OUT / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Validation Report",
        "",
        f"## Overall Assessment: {assessment}",
        "",
        "## Calculation Spot-Checks",
        "",
    ]
    for name, item in checks.items():
        status = "Verified" if item["passed"] else "Discrepancy found"
        lines.append(f"- **{name}: {status}.** `{json.dumps(item['evidence'], ensure_ascii=False)}`")
    lines += ["", "## Required Caveats for Stakeholders", ""]
    lines.extend(f"- {item}" for item in report["required_caveats"])
    (OUT / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
