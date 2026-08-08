# Eight local Transformer experiments — final audit

## Verdict

The local-only goal passes its evidence audit. All eight preregistered items were
implemented or tested under the frozen engine, seat-balance and opponent-pool
protocol. Failed arms were rejected without using Kaggle as a selector.

The accepted local candidate is experiment 7:

- Agent: `behavior_cloning/agents/grimmsnarl_multideck_identity_h8_pre12_v1`
- Package SHA256: `fb448ada8a2a7dabeb09563c7a229cb2a5cc64558e4c2a9f8834f17942523bb9`
- 200 games per opponent: 67.50% macro, +5.92 percentage points versus control.
- Fresh 400 games per opponent: 65.33% macro, +3.75 percentage points versus control.
- Fresh-400 cells: Tetsutani 70.00%, biohack44 50.75%, romanrozen 75.25%.
- Both frozen gates pass; engine errors and candidate fallbacks are zero.

The frozen `h8/pretrain-12` control is unchanged at package SHA256
`3c0ce5024e94f429dbbef8eb9c544deec2a953aea358e933624a78447ad26523`
and 61.58% macro over its 200-game-per-opponent matrix.

## Experiment decisions

| # | Experiment | Decision | Decisive evidence |
|---|---|---|---|
| 1 | Structured opponent public-event history | Rejected at 200 | 62.67% macro; +1.08pp, below +3pp gate. |
| 2 | Exact remaining-deck state and draw probabilities | Rejected near miss at 200 | 64.00% macro; +2.42pp, below +3pp gate. |
| 3 | Real-loss engine branch-search Q data | Data generation complete; not a direct override | 23 loss states x 64 paired terminal samples; replicate Pearson 0.853, sign agreement 52.17%; only one strict positive. |
| 4 | Action-value / advantage head | Rejected fail-closed | No strict positive in calibration, no safe takeover threshold; holdout remained sealed. |
| 5 | Frozen multi-deck, multi-opponent validation pool | Accepted infrastructure | Three promotion opponents with three exact deck hashes; 20 -> 200 -> fresh 400 protocol. |
| 6 | Eight-environment clipped PPO | Rejected at 200 | 65.17% macro, but Tetsutani harm was -5.75pp and macro regressed 2.33pp from experiment 7. |
| 7 | Exact own-deck encoding and opponent evidence | Policy accepted; auxiliary identifier rejected | 200 and fresh-400 gates pass. Classifier generalization failed on C20, so it is diagnostic only. |
| 8 | Joint deck and policy search | Rejected against current best | Selected legal hybrid scored 67.00% at 200, trailing experiment 7 by 0.50pp; no 400 run. |

Only the best passing arm received the independent 400-game replication, as
frozen before results. Offline-only failures were not promoted to arena play.

## Evidence boundary

- Validation protocol: `behavior_cloning/VALIDATION_POOL_PROTOCOL.md`
- Frozen pool: `behavior_cloning/validation_pool_manifest.json`
- Control matrix: `behavior_cloning/validation_control_matrix_200.json`
- Experiment receipts: the eight `experiment_result.json` files verified by
  `behavior_cloning/verify_transformer_8_experiment_goal.py`.
- Final verifier checks the live package receipts, 17 referenced artifact hashes,
  all eight statuses, the 20-game screen, both promotion matrices and both gates.
- Kaggle submission authorization is false for every experiment. No Kaggle
  submission was performed as part of this goal.

## Interpretation

The evidence supports a local promotion of experiment 7 over the frozen control,
not a claim about Kaggle leaderboard score or medal probability. The gain is
concentrated against the biohack44 matchup, but the fresh-400 Tetsutani and
romanrozen cells remain within the preregistered 5pp harm limit.
