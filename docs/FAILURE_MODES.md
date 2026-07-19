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
