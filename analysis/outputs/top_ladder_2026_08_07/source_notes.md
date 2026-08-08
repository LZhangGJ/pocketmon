# Source notes: top-ladder deck analysis

- Current source: `/homes/lzhang/pocketmon/data/raw/replays/2026-08-07` plus its `manifest.csv`.
- Comparison source: `/homes/lzhang/pocketmon/data/raw/replays/2026-08-06` plus its `manifest.csv`.
- Card mapping: official competition `EN Card Data.csv` and `JP Card Data.csv`.
- Elite cohort: top 10% of matches by `min_score`; current cutoff 1079.123.
- Analysis grain: one deck appearance per player per valid two-player replay.
- Archetypes: exact Pokémon-card multisets connected at IDF-weighted Jaccard >= 0.55; Trainer/Energy differences remain exact-deck variants.
- Win rate: reward > 0 is a win, reward = 0 is a half-win, reward < 0 is a loss. Non-mirror Wilson intervals are descriptive; repeated games by the same team are not independent.
- Score-to-player mapping is unavailable in the daily manifest; `min_score` is used so both players meet the cohort cutoff.
- Source timestamps have no timezone offset.

## Chart map

1. Meta share: ranked horizontal bar; archetype vs usage share; current elite cohort.
2. Day-over-day share: grouped bar; current vs previous elite usage share; top current archetypes.
3. Performance vs presence: scatter; usage share vs non-mirror win rate with sample size retained in the source table.
4. Card inclusion: ranked horizontal bar; top card inclusion rates; current elite cohort.
