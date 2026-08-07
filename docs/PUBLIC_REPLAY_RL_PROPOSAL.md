# Public replay RL proposal

## Status

This proposal is intentionally staged. Public replay labels must pass DATA-001 and DATA-002 before they are used for policy training. Public replay validation accuracy is not a substitute for gameplay evaluation against the fixed opponent snapshot.

## Data contract

Each policy sample contains only the acting player's observation at decision time, the complete legal option list, the selected option indices, final outcome, teacher/manifest metadata, and an immutable source hash. `observation.logs` is excluded from model input because it may contain privileged or post-decision information. Both player views must never be merged into a policy state.

The policy target receives non-zero weight only for identified strong teachers on winning trajectories. Losing and draw trajectories remain useful for the value function, failure classification, and conservative offline objectives. Splits are grouped by episode and should also hold out complete teacher submissions and dates to measure generalization rather than memorization.

## Recommended training stages

### Stage A — masked behavior cloning

- Resolve every option's semantic identity from `area/index`, including card, Pokemon, attack, energy, target, and selection context.
- Encode card IDs and categorical fields with embeddings rather than treating IDs as ordinal numbers.
- Use a public-board encoder plus hand/discard encoders and a recurrent history state.
- Score the variable legal-option set with a pointer/candidate network.
- Train multi-selection prompts autoregressively with a mask and an explicit stop action.
- Weight only high-confidence winning teacher decisions in the policy loss; retain every valid terminal outcome for the value head.

Promotion gate: zero invalid actions, improved held-out teacher action likelihood, and a statistically significant paired-seat win-rate gain over `lucario_rule` across the immutable public-opponent snapshot.

### Stage B — advantage-weighted offline RL

After the behavior policy is stable, fit twin value/Q heads and apply an offline method that stays close to the dataset support. Advantage-Weighted Regression or Implicit Q-Learning is preferred over unconstrained offline Q-learning because public agents are heterogeneous and many actions are suboptimal. Cap advantage weights and retain a behavior-cloning regularizer to reduce extrapolation error.

### Stage C — recurrent PPO self-play

Initialize ten independent learners from the best Stage-B checkpoint. Run recurrent PPO with legal-action masking, GAE, entropy regularization, gradient clipping, and a rule fallback on invalid/low-confidence decisions. Sample fixed public, champion, historical, and hard-counter opponents according to `configs/league_v1.json`. Freeze evaluation checkpoints; training games never count toward promotion.

### Stage D — league exploiters and distillation

After all ten learners pass every fixed public matchup, add best-response exploiters against payoff-matrix weaknesses while retaining historical opponents to prevent cycling. Distill the robust policy ensemble into a smaller submission model only after latency, memory, package-size, and regression gates pass.

## Required ablations

1. Rule Agent vs behavior cloning.
2. Winning-teacher filtering vs all-action imitation.
3. Card/option semantic embeddings vs current numeric IDs.
4. Feed-forward vs recurrent history encoder.
5. BC vs BC + offline RL.
6. Offline model vs PPO fine-tuning.
7. Neural-only vs neural policy with rule fallback.

Every comparison uses common seeds with swapped seats. Near-threshold candidates require 3,000–5,000 confirmatory games, in addition to the 400-game screening gate.
