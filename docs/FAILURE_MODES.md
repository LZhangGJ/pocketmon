# Failure modes

- **Opponent drift:** mutable notebooks invalidate comparisons. Mitigation: dated content-hashed snapshots.
- **Malicious notebook code:** public code may access secrets or damage the host. Mitigation: secretless, network-disabled container and resource limits before execution.
- **Leaderboard-selection bias:** public score may be noisy or stale. Mitigation: local smoke gate and fixed payoff-matrix evaluation.
- **Seat/RNG bias:** unpaired games exaggerate differences. Mitigation: common seeds with both seats.
- **Co-evolution cycling:** agents forget older counters. Mitigation: champion and historical snapshot archive.
- **False promotion:** raw win rate alone is noisy. Mitigation: Wilson lower bound, zero functional failures, confirmatory games near threshold.
- **Weak imitation:** mixing random or losing actions corrupts behavior cloning. Mitigation: teacher identity and outcome filtering.
- **Homogeneous collapse:** identical learners converge together. Mitigation: diversify seeds, entropy and opponent sampling; monitor policy/payoff diversity.
