# Failure modes

- **Opponent drift:** mutable notebooks invalidate comparisons. Mitigation: dated content-hashed snapshots.
- **Malicious notebook code:** public code may access secrets or damage the host. Mitigation: secretless, network-disabled container and resource limits before execution.
- **Leaderboard-selection bias:** public score may be noisy or stale. Mitigation: local smoke gate and fixed payoff-matrix evaluation.
- **Seat/RNG bias:** unpaired games exaggerate differences. Mitigation: common seeds with both seats.
- **Co-evolution cycling:** agents forget older counters. Mitigation: champion and historical snapshot archive.
- **False promotion:** raw win rate alone is noisy. Mitigation: Wilson lower bound, zero functional failures, confirmatory games near threshold.
- **Weak imitation:** mixing random or losing actions corrupts behavior cloning. Mitigation: teacher identity and outcome filtering.
- **Homogeneous collapse:** identical learners converge together. Mitigation: diversify seeds, entropy and opponent sampling; monitor policy/payoff diversity.
- **Replay step misalignment:** an action paired with its post-action observation creates impossible labels. Mitigation: DATA-001 compares candidate alignments and blocks conversion below the legal-action gate.
- **Privileged-information leakage:** merging both player views or using event logs can expose hidden/future information. Mitigation: one acting-player view per row and `observation.logs` removed by default.
- **Duplicate or mutable episodes:** repeated IDs can leak across splits or silently change labels. Mitigation: episode-level deduplication by ID and SHA-256; conflicting hashes fail DATA-002.
- **Teacher contamination:** weak, losing, crashed, or unidentified agents become policy targets. Mitigation: zero policy weight outside accepted winning teachers; keep other rows only for value/failure learning.
- **Unsafe unseen-state fallback:** a majority table may emit an action that is illegal in a new state. Mitigation: no global action default; fall back to the validated rule Agent.
- **Post-action status mistaken for submission status:** replay `steps[t].status` is the state after the interpreter processes `steps[t].action`; filtering it can select the next player and make valid actions look like empty placeholders. Mitigation: decide whether an action was requested using `steps[t-1].status`, retain all three status positions in reports, and test a turn-switch fixture where the two statuses differ.
- **Non-turn placeholder entries miscounted as decisions:** inactive agents can have empty/default action values in a replay step. Mitigation: skip them only when the previous-step submission status is not `ACTIVE`, while reporting initial, inactive, and unknown-status skip counts separately. DATA-002 fails on unknown non-initial status instead of silently filtering it.
- **Empty required action may be stored at another lag:** an empty action paired with `minCount >= 1` could indicate a wrapper-specific delay, repeated selection, or a remaining schema error. Mitigation: `diagnose_replay_action_positions.py` compares lags and records candidate actions/statuses/select fingerprints. Other lags are diagnostic evidence only; conversion remains blocked until strict DATA-001 passes.
- **Stale diagnostic evidence:** the first `status == ACTIVE` rerun used the status beside the action, so its 0.8917 rate and residual type-3/type-14 empty-action counts are contaminated by the status-position bug. Mitigation: retain the original failed report for provenance, but do not treat those residual counts as current facts until the corrected server rerun.
