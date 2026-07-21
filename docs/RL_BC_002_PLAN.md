# RL-BC-002: training-length and causal-history comparison

## Frozen reference

RL-BC-001 is frozen and must not be overwritten. Its formal three-seed reference uses batch size 256 and 30 epochs. Mean validation sequence exact, set exact, optional-empty accuracy, and multi-select accuracy are 0.448626, 0.454818, 0.354701, and 0.473545. Every decoded action was legal and the invalid-action count was zero.

## Schema and causality decision

The full schema-v2 gzip was audited before implementing history. All 78,776 rows contain `episode_id`, `player`, `action_step`, and `observation_step`. Grouping by `(episode_id, player)` produces 1,000 player trajectories with no duplicate explicit order key. Every row has `action_step - observation_step == 1`, and `player` agrees with `observation.current.yourIndex` on every row.

History must be constructed by sorting each `(episode_id, player)` group on `action_step`; physical JSONL order is never an ordering input. A row may consume only tokens built from rows with a strictly smaller `action_step`. Each token contains the prior row's pre-action visible-state encoding and a summary of the options selected by that prior action. It excludes the current action, all future rows, outcome/reward, winner, post-action state, logs, and newly introduced hidden information. The minimal GRU arm uses the latest 16 valid prior decisions; truncation is causal and is part of the fingerprinted configuration.

The machine-readable evidence is `results/rl_bc_002_history_audit.json`. Any future dataset that lacks these explicit keys, contains duplicate order keys, violates pre-action lag, or disagrees on player identity must block history training instead of falling back to file order.

## Formal arms

Arm A, `RL-BC-002-A` / stateless-long, retains the RL-BC-001 architecture and changes only the training budget to 60 epochs with patience 10. Arm B, `RL-BC-002-B` / history-encoder, uses the same data, split, seeds, batch size, learning rate, hidden size, losses, 60-epoch limit, and patience, adding only the causal GRU described above. Configurations are in `configs/rl_bc_002_stateless_long.json` and `configs/rl_bc_002_history_gru.json`.

Both arms use seeds 17, 42, and 20260720, batch size 256, learning rate 0.0003, hidden dimension 128, value-loss weight 0.25, and gradient-clip norm 1.0. Every seed starts from random initialization; RL-BC-001 checkpoints are never resumed. Formal runs require a clean worktree and record code commit, input SHA, full configuration, experiment fingerprint, checkpoint SHA, runtime, RAM, and VRAM. Fingerprints from different arms, commits, or configurations may not be aggregated.

## Gates and interpretation

Each arm requires all three seeds before a completed-formal claim. Validation decode legality must be 1.0 with zero invalid actions, unsupported rows, skipped rows, NaN, OOM, and runtime failures. Reports include per-seed metrics, select type/context groups, best epoch, late-epoch loss trend, and aggregate mean, sample standard deviation, minimum, and maximum.

The comparison asks whether training beyond epoch 30 helps the frozen stateless baseline and whether history adds an independent gain at the same 60-epoch budget. Direction should agree in at least two of three seeds. Offline improvements do not imply gameplay strength. A checkpoint-to-agent adapter and local game smoke are allowed only after both arms complete, and AWR, IQL, PPO, self-play, league training, and opponent-pool updates remain out of scope.
