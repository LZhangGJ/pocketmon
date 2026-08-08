# Pocketmon compatibility notes

This directory began as the teammate's Experiment 7 code-review snapshot. The model architecture, token organization, losses, portable inference and validation semantics are retained.

Two narrow training compatibility edits were applied:

1. Module-version strings absent from the historical `MODULE_WEIGHTS` table receive weight `1.0`; known historical versions retain their original weights.
2. Opponent auxiliary classes with zero visible-evidence examples in the selected fit split receive zero class-loss weight instead of aborting the main policy run.

These changes do not add runtime inputs, do not expose hidden opponent information and do not enable opponent-class runtime gating. Pocketmon replay conversion, deck selection, orchestration and packaging live outside this directory in `experiment7/integration/`.
