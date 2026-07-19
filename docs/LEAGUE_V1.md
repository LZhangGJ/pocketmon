# League-v1 experiment

League-v1 starts ten independently seeded learners against a versioned pool of strong public agents. Public notebooks are teachers and regression opponents, not mutable training dependencies.

## Daily opponent refresh

1. Export a curated JSON list of current high-scoring notebook slugs (`name`, `source`). Leaderboard association must be reviewed because Kaggle's notebook listing is not a reliable score oracle.
2. On the training server run `python scripts/refresh_opponent_pool.py --candidates configs/public_candidates.json`.
3. Quarantined downloads must contain `main.py` and `deck.csv`, compile, satisfy the size guard, then pass smoke games before `current.json` is advanced manually.
4. Never overwrite a dated snapshot. Record the snapshot manifest and hashes with every experiment.

Downloaded notebook code is untrusted. Refresh must run in an isolated account/container without secrets or network access during validation and games. Static compilation is only the first gate.

## Training loop

- Generation 0: clone the current policy into 10 learners, varying seed, entropy coefficient and public-opponent sampling.
- Warm start: behavior cloning only from winning trajectories of identified strong teachers. Do not mix random/losing actions into the policy target.
- RL: use recurrent PPO with legal-action masking. Sample public, champion, historical and hard-counter opponents according to the checked-in config.
- Freeze a checkpoint and run the full paired evaluation schedule; training games never count as evaluation.
- A learner passes only if every fixed public matchup clears both raw win rate and Wilson lower-bound gates with no crash, timeout or illegal action.
- Evolution starts only when all 10 learners pass. The next population is seeded from the qualified set, retains historical snapshots, and adds exploiters against weaknesses in the payoff matrix.

`games_per_pair=400` is a screening gate. Near-threshold candidates require a larger confirmatory run (normally 3,000–5,000 games per decisive comparison) before a PR or champion recommendation.

## Server commands

```bash
python -m unittest tests/test_league_v1.py
python scripts/refresh_opponent_pool.py --candidates configs/public_candidates.json --date YYYY-MM-DD
python scripts/league_v1.py --snapshot data/opponents/snapshots/YYYY-MM-DD/manifest.json --schedule results/league_v1_schedule.csv
# The match workers consume the schedule and write: learner,opponent,seed,learner_seat,result,latency_ms,memory_mb
python scripts/league_v1.py --results results/league_v1_games.csv --report results/league_v1_report.json
```

The current branch supplies orchestration and promotion gates; it does not claim completed server training or gameplay results.
