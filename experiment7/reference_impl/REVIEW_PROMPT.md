# Prompt for GPT

Please perform a source-grounded code review of this Pokemon TCG AI Battle
Experiment 7 implementation. Focus on defects that can change win rate or make
the Kaggle runtime fail.

Review in this priority order:

1. Hidden-information leakage: prove whether runtime features use only actor-visible
   opponent information and the agent's own legal deck information.
2. Training/runtime parity: compare feature construction, token ordering, masks,
   history updates, Transformer math and stable legal-action ranking.
3. Label/split correctness: episode isolation, chronological calibration/holdout,
   multi-select semantics, action-count targets and duplicate leakage risks.
4. Portable NumPy inference: compare it with the PyTorch model, especially layer
   norm, attention masks, GELU, padding and float behavior.
5. Kaggle CPU reliability: imports, global initialization, memory, latency,
   deterministic state reset, forced actions, error handling and fallback behavior.
6. Model/training design: deck multiset encoding, visible-opponent aggregation,
   auxiliary classification loss and possible overfitting or distribution-shift
   failure modes.
7. Validation validity: seat balance, process isolation, gate calculation and any
   way the reported local improvement could be overstated.

For every finding, cite the exact file and line, explain the concrete failure
mode, assign severity, and propose the smallest safe fix. Separate confirmed bugs
from hypotheses. Do not infer behavior from excluded weights or data; state what
cannot be verified from this code-only archive.

